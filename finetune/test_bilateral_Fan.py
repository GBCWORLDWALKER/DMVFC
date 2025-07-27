from __future__ import print_function, division

import argparse
from corr import get_corr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import re
import training_functions_fiber_pair
from utils.calculate_index_pair import generate_unique_pairs
from utils.distance_prepare import distance_prepare
from utils import utils, nets
from utils.utils import str2bool, params_to_str
from utils.io import process_args_test, read_data, output_data
from utils.clusters import metrics_calculation, outlier_removal, metrics_subject_calculation
import utils.fibers as fibers
import scipy.io as sio
import numpy as np
import os, h5py

if __name__ == "__main__":

    # Translate string entries to bool for parser
    parser = argparse.ArgumentParser(description='Use DFC for clustering')

    parser.add_argument('-inputDirectory',action="store", dest="inputDirectory", default=None, help='input SWM tractography (.vtp or .vtp)')
    parser.add_argument('-outdir', action="store", dest="outputDirectory",default=None, help='Output folder of clustering results.')
    parser.add_argument('-trained_net', default=None, help='path of trained net')
    parser.add_argument('-atlas_pred', default=None, help='path of pred net')
    parser.add_argument('-surf', default=True, type=str2bool, help='inporparating cortical information')
    parser.add_argument('-batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('-p', action="store", dest="numberOfFiberPoints", type=int, default=14, help='Number of points in each fiber to process. 14 is default.')
    parser.add_argument('-ro_std', default=1, type=int, help='outlier removal threshold')
    parser.add_argument('-output_clusters', default=True, type=str2bool, help='output cluster vtk files')
    parser.add_argument('-funct', default=True, type=str2bool, help='function pairs or not')
    parser.add_argument('-gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--wandb_group', default=None, type=str, help='wandb group')
    parser.add_argument('-fmri_similarity_path', default=None, type=str, help='path of fmri similarity')
    parser.add_argument('-dmri_similarity_path', default=None, type=str, help='path of dmri similarity')
    parser.add_argument('-epochs_pretrain', default=None, type=int, help='epochs of pretrain')
    parser.add_argument('--multi_view_test', default=None, help='multi-view test')
    parser.add_argument('--trained_net_b', default=None, help='path of trained net b')
    parser.add_argument('--bundle', choices=['CST_left','CST_right','AF_left','AF_right','CC_4','CC_5'], help='bundle name')
    parser.add_argument('--num_clusters', default=None, type=int, help='number of clusters')
    parser.add_argument('--not_pre_saved_results', default=False, type=str2bool, help='not pre-saved results')
    args = parser.parse_args()
    params = process_args_test(args)

    utils.print_both(params['log'], "===== SWM-DFC ======")
    utils.print_both(params['log'], params_to_str(params))

    # Data preparation
    utils.print_both(params['log'], "Reading data from %s " % params["inputDirectory"])
    input_pd, fiber_array = read_data(params, verbose=False)

    utils.print_both(params['log'], "Generating dataset")
    unique_subIDs, counts = torch.unique(torch.tensor(fiber_array.fiber_subID), return_counts=True)
    unique_subIDs=unique_subIDs.tolist()
    subID_counts = dict(zip(unique_subIDs, counts.tolist()))
    index_pair = dict()
    for subID in unique_subIDs:
        vec=subID_counts[subID]
        index_pair[subID] = generate_unique_pairs(vec, 2,seed=1)

    def numsort(x):
            num=re.findall(r'\d+',x)
            return int(num[0])
    fmri_path_list = [os.path.join(params['multi_view_test'], subject,fmri_path) 
                    for subject in sorted(os.listdir(params['multi_view_test']), key=numsort) for fmri_path in sorted(os.listdir(os.path.join(params['multi_view_test'],subject))) 
                    if params['bundle'] in fmri_path]
    fmri_array=[]
    for fmri_path in fmri_path_list:
        mat=sio.loadmat(fmri_path)
        fmri_array.append(mat['h'].transpose(2,1,0))
    fmri_array=np.concatenate(fmri_array,axis=0)
    dataset_b=fibers.FiberPair(fmri_array,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]),funct=params['funct'])

    dataset = fibers.FiberPair(fiber_array.fiber_array_ras,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]),funct=params['funct'])
    params['dataset_size'] = len(dataset)

    device = torch.device("cuda:"+str(params['gpu']) if torch.cuda.is_available() else "cpu")
    utils.print_both(params['log'], "\nPerforming calculations on:\t" + str(device))
    params['device'] = device
    if params['not_pre_saved_results']:
    # GPU check


        # Create DGCNN model
        weights = torch.load(params['trained_net'])
        num_clusters = weights['clustering.weight'].size()[0]
        embedding_dimension = weights['clustering.weight'].size()[1]
        idx = nets.get_dgcnn_idx(5, params["num_points"], params["batch_size"]).to(device)

        model = nets.DGCNN(k=5, input_channel=3, num_clusters=num_clusters, embedding_dimension=embedding_dimension, idx=idx,device=device)
        model = model.to(device)
        model.load_state_dict(weights)

        

        if params['multi_view_test'] is not None:
            weights_b = torch.load(params['trained_net_b'])
            num_clusters_b = weights_b['clustering.weight'].size()[0]
            embedding_dimension_b = weights_b['clustering.weight'].size()[1]
            idx_b = nets.get_dgcnn_idx(5, params["num_points"], params["batch_size"]).to(device)
            model_b = nets.DGCNN(k=5, input_channel=30, num_clusters=num_clusters_b, embedding_dimension=embedding_dimension_b, idx=idx_b,device=device)
            model_b = model_b.to(device)
            model_b.load_state_dict(weights_b)
            preds_final, probs_final =  training_functions_fiber_pair.test_model_multi(model,model_b, dataset,dataset_b, params)
        else:
            preds_final, probs_final = training_functions_fiber_pair.test_model(model, dataset, params)
    # output result
    else:
        num_clusters=params['num_clusters']
        preds_final=np.load(os.path.join(params['presaved_path'], f'preds.npy'),allow_pickle=True)
        probs_final=np.load(os.path.join(params['presaved_path'], f'probs.npy'),allow_pickle=True)
    print("\n=======================")
    utils.print_both(params["log"], "\n# Outlier removal:")
    params["num_clusters"] = num_clusters
    rejected_fibers = outlier_removal(params, preds_final, probs_final)
    fmri_array = np.delete(fmri_array, rejected_fibers, axis=0)
    preds_final = np.delete(preds_final, rejected_fibers, axis=0)
    # Store different clusters' fMRI arrays
    # Create a dictionary to store fMRI arrays for each subject and cluster
    dmri_array_id = np.delete(fiber_array.fiber_subID, rejected_fibers, axis=0)
    subject_cluster_fmri_arrays = {}

    # Iterate through unique subject IDs
    for subject_id in np.unique(dmri_array_id):
        subject_mask = dmri_array_id == subject_id
        subject_cluster_fmri_arrays[subject_id] = {}

        for cluster in range(num_clusters):
            cluster_mask = (preds_final == cluster) & subject_mask
            subject_cluster_fmri_arrays[subject_id][cluster] = fmri_array[cluster_mask]

    # Save the subject-specific cluster fMRI arrays
    for subject_id, cluster_data in subject_cluster_fmri_arrays.items():
        subject_fmri_clusters_h5 = os.path.join(params['outputDirectory'], f"{params['bundle']}_subject_{subject_id}_fmri_clusters.h5")
        with h5py.File(subject_fmri_clusters_h5, "w") as f:
            for cluster, cluster_fmri in cluster_data.items():
                f.create_dataset(f'cluster_{cluster}', data=cluster_fmri)
        utils.print_both(params['log'], f"\nSaved fMRI clusters for subject {subject_id} to:\t{subject_fmri_clusters_h5}")
    get_corr(params['outputDirectory'])
    output_dir = params['outputDirectory']
    os.makedirs(output_dir, exist_ok=True)
    preds_h5 = os.path.join(output_dir, params['subID']+"_pred.h5")
    with h5py.File(preds_h5, "w") as f:
        f.create_dataset('preds_final', data=preds_final)
        f.create_dataset('probs_final', data=probs_final)
        f.create_dataset('rejected_fibers', data=rejected_fibers)
    utils.print_both(params['log'], "\nSave pred :\t" + preds_h5)

    utils.print_both(params["log"], "\n# Evalution metrics for the atlas:")
    params['output_name'] = "SMW"
    params['output_dir'] = params['outputDirectory']
    df_stats, cluster_centroids, cluster_reordered_fibers = \
        metrics_calculation(params, fiber_array, preds_final, probs_final, outlier_fibers=rejected_fibers, output=True, verbose=True)

    # output centroid and cluster vtk files
    utils.print_both(params["log"], "\n# Output clustering results:")
    surf_cluster = None
    output_pd = output_data( fiber_array.fiber_subID,params, input_pd, preds_final, probs_final, surf_cluster, df_stats, cluster_centroids, cluster_reordered_fibers, outlier_fibers=rejected_fibers, clusters_to_remove=[], verbose=False)

    # Close files
    params["log"].close()


# export CUDA_VISIBLE_DEVICES=6; /home/fan/Software/miniconda3/py310_23.3.1-0/envs/SWM-DFC/bin/python \
# 	/home/fan/Projects/SWM_DFC/test_bilateral_Fan.py \
# 	-inputvtk /data02/SWM-HCP-S1200/tractography/MNI-SWM-n1065/individual-surf/100408-regionfiltered-HOAlabeled-lmin0-lmax100-umin0-umax95-full-SurfDis.vtp \
# 	-outdir /data01/fan/SWM-DFC/data/SWM_f20k_n1065/EmbedDisSurf_LossDisSurf/K4000/it3/subjects/ \
# 	-trained_net /data01/fan/SWM-DFC/data/SWM_f20k_n1065/EmbedDisSurf_LossDisSurf/K4000/it3/models/DGCNN_001_final_k4000.pt \
# 	-atlas_pred /data01/fan/SWM-DFC/data/SWM_f20k_n1065/EmbedDisSurf_LossDisSurf/K4000/it3/DFC_it3_DGCNN_001_EmbEPTrue_CluEPTrue_gamma0.1_ro1_k4000/DFC_it3_DGCNN_001_EmbEPTrue_CluEPTrue_gamma0.1_ro1_k4000_pred.h5 \
# 	-p 30 -surf True