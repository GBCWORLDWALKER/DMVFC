from __future__ import print_function, division
import datetime
import random
import logging
import sys
import time
import glob
import shutil
import argparse
import multiprocessing
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import h5py
import whitematteranalysis as wma
import scipy.io as sio
import yaml
import copy # 新增导入 for deepcopy
import training_functions_fiber_pair
from utils.distance_prepare import distance_prepare,distance_prepare_tract,distance_prepare_single_tract
from utils import utils, nets
from utils.utils import str2bool, params_to_str
from utils.io import process_args, read_data, output_data
from utils.clusters import metrics_calculation, outlier_removal, metrics_subject_calculation
import utils.fibers as fibers
from utils.calculate_index_pair import generate_unique_pairs

from cnslab_fmri.net.st_gcn import st_gcn_fiber

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        scripts_path = os.path.join(path, 'scripts')
        if not os.path.exists(scripts_path):
            os.mkdir(scripts_path)
        for script in scripts_to_save:
            dst_file = os.path.join(scripts_path, os.path.basename(script))
            shutil.copyfile(script, dst_file)

def load_config(args):
    if args.config:
        # Load YAML configuration
        yaml_config = yaml.load(open(args.config, encoding="utf-8"), Loader=yaml.FullLoader)
        # Update the command-line arguments with YAML config
        for key, value in yaml_config.items():
            setattr(args, key, value)
        # Convert the updated dictionary back to an argparse Namespace
 
    else:
        args = vars(args)
    return args

def main_train(params,args,writer):
    utils.print_both(params['log'], "===== SWM-DFC ======")
    utils.print_both(params['log'], params_to_str(params))
    utils.print_both(params['log'], "# Run name: \t %s" % params["run_name"])

    # Data preparation
    utils.print_both(params['log'], "Reading data from %s " % params["inputDirectory"])
    
    input_pd, fiber_array = read_data(params, verbose=False)
    utils.print_both(params['log'], "Generating dataset, with embedding_surf = %s" % params['embedding_surf'] )
    vec=len(fiber_array.fiber_subID==101107)
    epoch = params['epochs_pretrain']+params['epochs']*2

    #get index pairs
    # Get unique elements of fiber_array.fiber_subID
    # 
    unique_subIDs, counts = torch.unique(torch.tensor(fiber_array.fiber_subID), return_counts=True)
    unique_subIDs=unique_subIDs.tolist()
    subID_counts = dict(zip(unique_subIDs, counts.tolist()))
    
    
    # if params['dataset_prepared']==False:
    #     index_pair = dict()
    #     for subID in unique_subIDs:
    #         vec=subID_counts[subID]
    #         index_pair[subID] = generate_unique_pairs(vec, epoch,seed=1)
            
    #     index_pair_file = os.path.join(params['outputDirectory'], 'index_pair.pkl')
    #     with open(index_pair_file, 'wb') as f:
    #         pickle.dump(index_pair, f)
    #     utils.print_both(params['log'], f"Saved index_pair to {index_pair_file}")
    # Load index_pair from the saved file
    # else:

    index_pair=torch.load(params['index_pair_path']).unsqueeze(1).expand(-1, 80, -1, -1).flatten(start_dim=1,end_dim=-2).cpu().numpy()

    

    # prepare dataset
    print("----------dataset preparing----------")
    
    def numsort(x):
        num=re.findall(r'\d+',x)
        return int(num[0])
    fmri_path_list = [os.path.join(params['fmri_path'], subject) 
                      for subject in sorted(os.listdir(params['fmri_path']), key=numsort) 
                      if params['bundle'] in subject][:80]
    fmri_array=[]
    
    for fmri_path in fmri_path_list:
        f=sio.loadmat(fmri_path)
        fmri_array.append(f['W_all'])
    fmri_array=np.concatenate(fmri_array,axis=0)
    
    if not params['dataset_prepared']:
        distance_prepare_single_tract(index_pair,subID_counts, fiber_array.fiber_array_ras, params['embedding_surf'],epoch,params['dmri_similarity_path'],params['bundle'])

    
    print("----------dataset preparation finished----------")

    if params['model_b']=="dgcnn":
        fmri_array=fmri_array[:,[0,-1],::2].reshape(-1,40,30)#(batch,25,1200)

    
    dataset_b=fibers.FiberPair(
        fmri_array,
        fiber_array.fiber_subID,
        index_pair,
        subID_counts,
        fmri_similarity_path=params['fmri_similarity_path'],
        dmri_similarity_path=None,
                               transform=transforms.Compose([transforms.ToTensor()]),
                               embedding_surf=params['embedding_surf'],
                               funct=params['funct'],
                               bundle=params['bundle'],model_b=params['model_b'])#(num,15,3)
    params['dataset_size'] = len(dataset_b)

    device = torch.device("cuda:"+str(params['GPU']) if torch.cuda.is_available() else "cpu")
    utils.print_both(params['log'], "\nPerforming calculations on:\t" + str(device))
    params['device'] = device



    if params['model_b']=="st"  :
        model_b =st_gcn_fiber(1, 1, embedding_dimension=10,num_clusters=params['num_clusters'])
    elif params['model_b']=="dgcnn":
        idx = nets.get_dgcnn_idx(10, 40, params["batch_size"]).to(device) if args.idx else None
        model_b = nets.DGCNN( k=10, input_channel=30, num_clusters=params["num_clusters"], embedding_dimension=10, idx=idx,device=device)
    else:
        raise ValueError(f"Invalid model_b: {params['model_b']}")

    #-----------------------------------model prepare-----------------------------
    model_b = model_b.to(device)
    num_pa = nets.count_parameters(model_b)
    utils.print_both(params['log'], "Number of b network parameters:\t" + str(num_pa))
    optimizer_b = optim.Adam(filter(lambda p: p.requires_grad, model_b.parameters()), lr=params["rate"], weight_decay=params["weight"])
    optimizer_pretrain_b = optim.Adam(filter(lambda p: p.requires_grad, model_b.parameters()), lr=params["rate_pretrain"], weight_decay=params["weight_pretrain"])
    optimizers_b = [optimizer_b, optimizer_pretrain_b]
    criterion_1 = nn.MSELoss(reduction='mean') # Reconstruction loss
    criterion_2 = nn.KLDivLoss(reduction='batchmean') # Clustering loss
    criteria = [criterion_1, criterion_2]
    scheduler_b = lr_scheduler.StepLR(optimizer_b, step_size=params["sched_step"], gamma=params["sched_gamma"])
    scheduler_pretrain_b = lr_scheduler.StepLR(optimizer_pretrain_b, step_size=params["sched_step_pretrain"], gamma=params["sched_gamma_pretrain"])
    schedulers_b = [scheduler_b, scheduler_pretrain_b]





#-----------------------------------model training-----------------------------

    model_b,preds_final, probs_final, surf_cluster = \
        training_functions_fiber_pair.train_model( model_b, dataset_b,criteria,optimizers_b, schedulers_b, params, fiber_array, writer)





#-----------------------------------model save-----------------------------
    final_net_save_b = params['model_files'][5]
    torch.save(model_b.state_dict(), final_net_save_b)
    utils.print_both(params['log'], "\nSave final model b:\t" + final_net_save_b)

    # output result
    print("\n=======================")
    utils.print_both(params["log"], "\n# Outlier removal:")
    rejected_fibers = outlier_removal(params, preds_final, probs_final)

    output_name = params['output_name']
    output_dir = params['output_dir']
    preds_h5 = os.path.join(output_dir, output_name+"_pred.h5")
    with h5py.File(preds_h5, "w") as f:
        f.create_dataset('preds_final', data=preds_final)
        f.create_dataset('probs_final', data=probs_final)
        f.create_dataset('rejected_fibers', data=rejected_fibers)
        if surf_cluster is not None:
            f.create_dataset('surf_cluster', data=surf_cluster.detach().cpu().numpy())
    utils.print_both(params['log'], "\nSave surf :\t" + preds_h5)

    temp = np.ones(len(preds_final))
    temp[rejected_fibers] = 0
    mask = temp > 0
    input_pd_retained = wma.filter.mask(input_pd, mask, verbose=False, preserve_point_data=True, preserve_cell_data=True)
    ouput_appedned_folder = os.path.join(output_dir, 'outlier_removed')
    os.makedirs(ouput_appedned_folder, exist_ok=True)
    pd_out_fname = os.path.join(ouput_appedned_folder, 'SWM_all_clusters.vtp')
    wma.io.write_polydata(input_pd_retained, pd_out_fname)
    utils.print_both(params['log'], "\nSave outlier removed tract:\t" + pd_out_fname)

    

    # utils.print_both(params["log"], "\n# Evalution metrics for the atlas:")
    # df_stats, cluster_centroids, cluster_reordered_fibers = \
    #     metrics_calculation(params, fiber_array, preds_final, probs_final, outlier_fibers=rejected_fibers, verbose=True)

    # # # subject-level stats
    # # utils.print_both(params["log"], "\n# Evalution metrics for each subject:")
    # # df_stats_subjects = metrics_subject_calculation(params, fiber_array, preds_final, probs_final, rejected_fibers, df_stats)

    # # probs = df_stats['Pred-Prob'].to_numpy().astype(np.float16)
    # # thr = probs.mean() - 2 * probs.std()
    # # clusters_to_remove = np.where(probs <= thr)[0]
    # # clusters_to_remove = []

    # # output centroid and cluster vtk files
    # utils.print_both(params["log"], "\n# Output clustering results:")
    # output_pd = x(params, input_pd, preds_final, probs_final, surf_cluster, df_stats, cluster_centroids, cluster_reordered_fibers, outlier_fibers=rejected_fibers, clusters_to_remove=[], verbose=False)

    # # Close files
    # params["log"].close()
    # if params['writer'] is not None:
    #     params['writer'].close()
from multiprocessing import Process

def train_model(i, model_b, j, args):
    args.sim_scale = i
    args.model_b = model_b
    args.GPU = j
    args.output_name = f"{args.output_name}_sim_scale_{i}_{model_b}"
    args.exp_name = f"{datetime.datetime.now().strftime('%m_%d_%H_%M_%S')}_{args.output_name}"

    exp_path = os.path.join(args.outputDirectory, args.exp_name)
    os.makedirs(exp_path, exist_ok=True)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")
    fh = logging.FileHandler(os.path.join(exp_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    with open(os.path.join(exp_path, "config.yml"), "w", encoding="utf-8") as f:
        yaml.dump(vars(args), f)

    writer = SummaryWriter(f"{args.outputDirectory}/{args.exp_name}/runs/{time.strftime('%m-%s', time.localtime())}-{random.randint(0, 100):05d}")

    create_exp_dir(exp_path, scripts_to_save=glob.glob('*.py'))

    args.outputDirectory = os.path.join(exp_path, "output")
    params = process_args(args)
    main_train(params, args, writer)
    
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Use DFC for clustering')
    parser.add_argument('--config', type=str, default="/data06/jinwang/isbi/src/DFC_multi_v3_bundle/config/train.yaml", help='Path to the config file', required=False)
    parser.add_argument('-indir', action="store", dest="inputDirectory", default=None, help='A folder of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument('-outdir', action="store", dest="outputDirectory", default=None, help='Output folder of clustering results.')
    parser.add_argument('-trf', action="store", dest="numberOfFibers_train", type=int, default=None, help='Number of fibers of each training data to analyze from each subject. None is all.')
    parser.add_argument('-l', action="store", dest="fiberLength", type=int, default=0, help='Minimum length (in mm) of fibers to analyze. 0mm is default.')
    parser.add_argument('-p', action="store", dest="numberOfFiberPoints", type=int, default=14, help='Number of points in each fiber to process. 14 is default.')
    parser.add_argument('-bold', action="store", dest="bold", type=str2bool, default=True, help='path to bold data')
    parser.add_argument('--embedding_surf', default=False, type=str2bool, help='incorporating cortical information')
    parser.add_argument('--reclustering', default=True, type=str2bool, help='reclustering')
    parser.add_argument('--loss_subID', default=False, type=str2bool, help='include sub ID in loss computation')
    parser.add_argument('--loss_surf', default=False, type=str2bool, help='include surf in loss computation')
    parser.add_argument('--ro', default=True, type=str2bool, help='outlier removal')
    parser.add_argument('--ro_std', default=1, type=int, help='outlier removal threshold')
    parser.add_argument('--num_clusters', default=2000, type=int, help='number of clusters')
    parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
    parser.add_argument('--clustering_fiber_interval', default=1, type=int, help='downsample of fibers for clustering')
    parser.add_argument('--epochs', default=1, type=int, help='clustering epochs')
    parser.add_argument('--epochs_pretrain', default=300, type=int, help='pretraining epochs')
    parser.add_argument('--output_clusters', default=True, type=str2bool, help='output cluster vtk files')
    parser.add_argument('--freeze', default=False, type=str2bool, help='freeze embedding training during clustering')
    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
    parser.add_argument('--pretrain', default=True, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--pretrained_net_a', default=None, help='index or path of pretrained net')
    parser.add_argument('--pretrained_net_b', default=None, help='index or path of pretrained net')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha for similarity')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--net_architecture', default='DGCNN', choices=['CAE_pair', 'DGCNN', 'PointNet', 'GCN'], help='network architecture used')
    parser.add_argument('--idx', default=True, type=str2bool, help='idx for dgcnn')
    parser.add_argument('--k', default=5, type=int, help='k for dgcnn')
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--rate', default=0.00001, type=float, help='learning rate for clustering')
    parser.add_argument('--rate_pretrain', default=0.0003, type=float, help='learning rate for pretraining')
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_step_pretrain', default=200, type=int, help='scheduler steps for rate update - pretrain')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float, help='scheduler gamma for rate update - pretrain')
    parser.add_argument('--printing_frequency', default=10, type=int, help='training stats printing frequency')
    parser.add_argument('--dataset_prepared', default=True, type=str2bool, help='dataset prepared')
    parser.add_argument('--update_interval', default=100, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--custom_img_size', default=[128, 128, 3], nargs=3, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    parser.add_argument('--full_brain', default=False, type=str2bool, help='full brain')
    parser.add_argument('--single_prepared', default=False, type=str2bool, help='single prepared')
    parser.add_argument('--funct', default=False, type=str2bool, help='full brain')
    parser.add_argument('--GPU', default=0, type=int, help='GPU number') # 确保 --GPU 参数存在
    parser.add_argument('--wandb_group', default=None, type=str, help='wandb group')
    parser.add_argument('-fmri_path', action="store", dest="fmri_path", type=str, default=None, help='path to fmri data')
    parser.add_argument('--fmri_similarity_path', action="store", dest="fmri_similarity_path", type=str, default=None, help='path to fmri similarity data')
    parser.add_argument('--dmri_similarity_path', action="store", dest="dmri_similarity_path", type=str, default=None, help='path to dmri similarity data')
    parser.add_argument('--strategy', default=0, type=int, help='strategy for loss computation')
    parser.add_argument('--strategy_initial', default=0, type=int, help='strategy for initial clustering')
    parser.add_argument('--loss_strategy', default=0, type=int, help='strategy for loss computation')
    parser.add_argument('--bundle', default='CST_left', choices=['CST_left', 'CST_right', 'AF_left', 'AF_right', 'CC_4', 'CC_5', 'SLF_II_left', 'SLF_II_right', "SLF_III_left", "SLF_III_right", "SLF_I_left", "SLF_I_right", "CC_1", "CC_2", "CC_3", "CC_6", "CC_7"], help='bundle')
    parser.add_argument('--output_name', default='SWM', type=str, help='output name')   
    
    args = parser.parse_args()

    # Load and merge YAML configuration
    # load_config 会修改传入的 args 对象，并返回它
    args_from_config = load_config(args) # args_from_config.GPU 现在拥有从YAML加载的GPU ID

    # list_1 原本硬编码了 GPU ID 为 0。
    # 现在我们使用从配置文件加载的 GPU ID (args_from_config.GPU)。
    # train_model 的第三个参数 'j' 被用作 GPU ID。
    # 这里的 i 和 model 分别对应 sim_scale 和 model_b 的示例值。
    # 如果这些也需要从配置或命令行读取，则需要进一步修改。
    # 目前，我们只修正 GPU ID。
    
    # 假设 sim_scale 和 model_b 的值是固定的，或者可以从 args_from_config 获取
    # 为了保持与原 list_1 结构相似，我们只替换 GPU ID 部分
    # 如果 sim_scale 和 model_b 也应该来自配置，可以这样：
    # sim_scale_to_use = getattr(args_from_config, 'sim_scale', 100) # 假设 sim_scale 在配置中
    # model_b_to_use = getattr(args_from_config, 'model_b', "dgcnn") # 假设 model_b 在配置中
    sim_scale_to_use = 100  # 保持原始示例值
    model_b_to_use = "dgcnn" # 保持原始示例值
    
    gpu_id_to_use = args_from_config.GPU # 使用从YAML加载的GPU ID

    list_of_runs = [(sim_scale_to_use, model_b_to_use, gpu_id_to_use)]
    
    processes = []
    for (current_sim_scale, current_model_b, current_gpu_id) in list_of_runs:
        # 为每个进程创建一个独立的args副本，以防train_model内部修改影响其他潜在的并行进程
        args_for_this_process = copy.deepcopy(args_from_config)
        
        p = Process(target=train_model, args=(current_sim_scale, current_model_b, current_gpu_id, args_for_this_process))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
