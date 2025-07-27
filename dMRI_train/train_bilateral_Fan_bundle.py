from __future__ import print_function, division

import argparse
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pickle
import training_functions_fiber_pair
from utils.distance_prepare import distance_prepare,distance_prepare_tract,distance_prepare_single_tract
from utils import utils, nets
from utils.utils import str2bool, params_to_str
from utils.io import process_args, read_data, output_data
from utils.clusters import metrics_calculation, outlier_removal, metrics_subject_calculation
import utils.fibers as fibers
from utils.calculate_index_pair import generate_unique_pairs
import numpy as np
import os
import h5py
import whitematteranalysis as wma
import nibabel as nib
if __name__ == "__main__":



    # Translate string entries to bool for parser
    parser = argparse.ArgumentParser(description='Use DFC for clustering')

    parser.add_argument('-indir',action="store", dest="inputDirectory", default=None, help='A folder of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument('-outdir',action="store", dest="outputDirectory",default=None, help='Output folder of clustering results.')
    parser.add_argument('-fmri_path', action="store", dest="fmri_path", type=str, default=None, help='path to fmri data')
    parser.add_argument('-trf', action="store", dest="numberOfFibers_train", type=int, default=None, help='Number of fibers of each training data to analyze from each subject. None is all.')
    parser.add_argument('-l', action="store", dest="fiberLength", type=int, default=0, help='Minimum length (in mm) of fibers to analyze. 0mm is default.')
    parser.add_argument('-p', action="store", dest="numberOfFiberPoints", type=int, default=14, help='Number of points in each fiber to process. 14 is default.')
    parser.add_argument('-bold',action="store", dest="bold", type=str2bool, default=True, help='path to bold data')
    parser.add_argument('--embedding_surf', default=False, type=str2bool, help='inporparating cortical information')
    parser.add_argument('--reclustering', default=True, type=str2bool, help='reclustering')
    parser.add_argument('--loss_subID', default=False, type=str2bool, help='include sub ID in loss computation')
    parser.add_argument('--loss_surf', default=False, type=str2bool, help='include surf in loss computation')
    parser.add_argument('--ro', default=True, type=str2bool, help='outlier removal')
    parser.add_argument('--full_brain', default=False, type=str2bool, help='full brain')
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
    parser.add_argument('--pretrained_net', default=None, help='index or path of pretrained net')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha for similarity')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--net_architecture', default='DGCNN', choices=['CAE_pair','DGCNN','PointNet','GCN'], help='network architecture used')
    parser.add_argument('--idx', default=True, type=str2bool, help='idx for dgcnn')
    parser.add_argument('--k', default=5, type=int, help='k for dgcnn')
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--rate', default=0.00001, type=float, help='learning rate for clustering')
    parser.add_argument('--rate_pretrain', default=0.0003, type=float, help='learning rate for pretraining')
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_step_pretrain', default=200, type=int,help='scheduler steps for rate update - pretrain')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,help='scheduler gamma for rate update - pretrain')
    parser.add_argument('--printing_frequency', default=10, type=int, help='training stats printing frequency')
    parser.add_argument('--dataset_prepared', default=True, type=str2bool, help='dataset prepared')
    parser.add_argument('--update_interval', default=100, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--custom_img_size', default=[128, 128, 3], nargs=3, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    parser.add_argument('--funct', default=False, type=str2bool, help='full brain')
    parser.add_argument('--GPU', default=0, type=int, help='GPU number')
    parser.add_argument('--wandb_group', default=None, type=str, help='wandb group')
    parser.add_argument('--bundle',default='CST_left',type=str,help='bundle')
    parser.add_argument('--similarity_path',default=None,type=str,help='similarity path')
    parser.add_argument('--fmri_similarity_path',default=None,type=str,help='fmri similarity path')
    parser.add_argument('--dmri_similarity_path',default=None,type=str,help='dmri similarity path')
    parser.add_argument('--output_name',default=None,type=str,help='output name')
    parser.add_argument('--bundle_list_path', type=str, default=None, 
                    help='Path to the directory containing bundle files')
    parser.add_argument('--index_path', type=str, default=None, 
                    help='Path to the directory containing bundle files')
    args = parser.parse_args()
    params = process_args(args)
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

    index_pair_file = os.path.join(params['outputDirectory'], 'index_pair.pkl')
    with open(index_pair_file, 'rb') as f:
        index_pair = pickle.load(f)
    utils.print_both(params['log'], f"Loaded index_pair from {index_pair_file}")
    
    def numsort(x):
        num=re.findall(r'\d+',x)
        return int(num[0])
    print("----------dataset preparing----------")
    distance_prepare(params['similarity_path'],params['bundle'], fiber_array.fiber_array_ras,params['fmri_similarity_path'],params['dmri_similarity_path'],epoch,params)
    print("----------dataset preparation finished----------")
    #define a new dataset with a shape of (num,25,3,30)

    
    dataset = fibers.FiberPair(fiber_array.fiber_array_ras,
                               fiber_array.fiber_subID,
                               index_pair,
                               subID_counts,
                               similarity_path=params['similarity_path'],
                               dmri_similarity_path=params['dmri_similarity_path'],
                               bundle=params['bundle'],
                               alpha=params['alpha'],
                               transform=transforms.Compose([transforms.ToTensor()]),
                               embedding_surf=params['embedding_surf'],
                               funct=params['funct'])#(num,15,3)
    
    params['dataset_size'] = len(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=False, num_workers=10)
    dataloader_entire = torch.utils.data.DataLoader(dataset, batch_size=int(params["batch_size"]/2), shuffle=False, num_workers=10)

    # GPU check
    device = torch.device("cuda:"+str(params['GPU']) if torch.cuda.is_available() else "cpu")
    utils.print_both(params['log'], "\nPerforming calculations on:\t" + str(device))
    params['device'] = device

    
    # Create DGCNN model
    idx = nets.get_dgcnn_idx(args.k, params["num_points"], params["batch_size"]).to(device) if args.idx else None
    model = nets.DGCNN(k=args.k, input_channel=3, num_clusters=params["num_clusters"], embedding_dimension=params["embedding_dimension"], idx=idx,device=device)
    model = model.to(device)

    num_pa = nets.count_parameters(model)
    utils.print_both(params['log'], "Number of network parameters:\t" + str(num_pa))

    # Prepare losses, optimizers and schedulers
    criterion_1 = nn.MSELoss(reduction='mean') # Reconstruction loss
    criterion_2 = nn.KLDivLoss(reduction='batchmean') # Clustering loss
    criteria = [criterion_1, criterion_2]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["rate"], weight_decay=params["weight"])
    optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["rate_pretrain"], weight_decay=params["weight_pretrain"])
    optimizers = [optimizer, optimizer_pretrain]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=params["sched_step"], gamma=params["sched_gamma"])
    scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=params["sched_step_pretrain"], gamma=params["sched_gamma_pretrain"])
    schedulers = [scheduler, scheduler_pretrain]

    # Train model
    model, preds_final, probs_final, surf_cluster = \
        training_functions_fiber_pair.train_model(model, dataset,criteria, optimizers, schedulers, params, fiber_array)

    final_net_save = params['model_files'][2]
    torch.save(model.state_dict(), final_net_save)
    utils.print_both(params['log'], "\nSave final model:\t" + final_net_save)

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

    exit()

    utils.print_both(params["log"], "\n# Evalution metrics for the atlas:")
    df_stats, cluster_centroids, cluster_reordered_fibers = \
        metrics_calculation(params, fiber_array, preds_final, probs_final, outlier_fibers=rejected_fibers, verbose=True)

    # # subject-level stats
    # utils.print_both(params["log"], "\n# Evalution metrics for each subject:")
    # df_stats_subjects = metrics_subject_calculation(params, fiber_array, preds_final, probs_final, rejected_fibers, df_stats)

    # probs = df_stats['Pred-Prob'].to_numpy().astype(np.float16)
    # thr = probs.mean() - 2 * probs.std()
    # clusters_to_remove = np.where(probs <= thr)[0]
    # clusters_to_remove = []

    # output centroid and cluster vtk files
    utils.print_both(params["log"], "\n# Output clustering results:")
    output_pd = x(params, input_pd, preds_final, probs_final, surf_cluster, df_stats, cluster_centroids, cluster_reordered_fibers, outlier_fibers=rejected_fibers, clusters_to_remove=[], verbose=False)

    # Close files
    params["log"].close()
    if params['writer'] is not None:
        params['writer'].close()

