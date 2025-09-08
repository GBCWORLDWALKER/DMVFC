from __future__ import print_function, division

import argparse
from cnslab_fmri.net.st_gcn import st_gcn_fiber
from corr_origin_quickbundles import get_corr as get_corr_origin_quickbundles
from corr_origin import get_corr as get_corr_origin
from corr_end import get_corr 
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
    parser.add_argument('--bundle', choices=['CST_left','CST_right','AF_left','AF_right','CC_4','CC_5','SLF_I_left','SLF_I_right','SLF_II_left','SLF_II_right','SLF_III_right','SLF_III_left','CC_1','CC_2','CC_3','CC_6','CC_7'], help='bundle name')
    parser.add_argument('--baseline', default=False, type=str2bool, help='baseline or not')
    parser.add_argument('--origin', default=False, type=str2bool, help='origin or not')
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
    def numsort(x):
        num=re.findall(r'\d+',x)
        return int(num[0])
    for subID in unique_subIDs:
        vec=subID_counts[subID]
        index_pair[subID] = generate_unique_pairs(vec, 2,seed=1)
    index_pair=index_pair[0]




    fmri_path_list = [os.path.join(params['fmri_path'], subject) 
                      for subject in sorted(os.listdir(params['fmri_path']), key=numsort) 
                      if params['bundle'] in subject][:20]
    fmri_array=[]
    
    for fmri_path in fmri_path_list:
        f=sio.loadmat(fmri_path)
        fmri_array.append(f['srvf_values'])
    fmri_array=np.concatenate(fmri_array,axis=0)

    


    fmri_array_1=fmri_array
    dataset=fibers.FiberPair(fmri_array_1,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]),funct=params['funct'])

    fmri_array_endpoint=fmri_array[:,:,::2]
    params['dataset_size'] = len(dataset)



    # GPU check
    device = torch.device("cuda:"+str(params['gpu']) if torch.cuda.is_available() else "cpu")
    utils.print_both(params['log'], "\nPerforming calculations on:\t" + str(device))
    params['device'] = device

    # Create DGCNN model
    weights = torch.load(params['trained_net'])
    num_clusters = weights['clustering.weight'].size()[0]
    embedding_dimension = weights['clustering.weight'].size()[1]


    
    model =  st_gcn_fiber(1, 1, embedding_dimension=embedding_dimension,num_clusters=params['num_clusters'])
    model = model.to(device)
    model.load_state_dict(weights)

    preds_final, probs_final = training_functions_fiber_pair.test_model(model, dataset, params)
    # output result
    print("\n=======================")
    utils.print_both(params["log"], "\n# Outlier removal:")
    params["num_clusters"] = num_clusters
    rejected_fibers = outlier_removal(params, preds_final, probs_final)

    fmri_array_endpoint_quickbundles = fmri_array_endpoint
    fmri_array_endpoint = np.delete(fmri_array_endpoint, rejected_fibers, axis=0)

    preds_final_1 = np.delete(preds_final, rejected_fibers, axis=0)
    # Store different clusters' fMRI arrays
    # Create a dictionary to store fMRI arrays for each subject and cluster
    dmri_array_id = np.delete(fiber_array.fiber_subID, rejected_fibers, axis=0)
    dmri_array_id_quickbundles = fiber_array.fiber_subID

    subject_cluster_fmri_arrays_endpoint = {}
    subject_cluster_fmri_arrays_endpoint_quickbundles = {}
    subject_cluster_quickbundles_arrays = {}
    # Iterate through unique subject IDs
    from dipy.segment.clustering import QuickBundles
    from dipy.segment.metric import AveragePointwiseEuclideanMetric

    # Convert fiber array to the format expected by QuickBundles
    streamlines = fiber_array.fiber_array_ras
    
    # Set up the clustering metric
    metric = AveragePointwiseEuclideanMetric()
    
    # Initialize QuickBundles with a distance threshold
    qb = QuickBundles(threshold=10.0, metric=metric,max_nb_clusters=num_clusters)
    
    # Perform the clustering
    clusters = qb.cluster(streamlines)
    
    for subject_id in np.unique(dmri_array_id):
        subject_mask = dmri_array_id == subject_id
        subject_mask_quickbundles = dmri_array_id_quickbundles == subject_id

        subject_cluster_fmri_arrays_endpoint[subject_id] = {}
        subject_cluster_fmri_arrays_endpoint_quickbundles[subject_id] = {}
        subject_cluster_quickbundles_arrays[subject_id] = {}
        for cluster in range(num_clusters):
            cluster_mask = (preds_final_1 == cluster) & subject_mask
            a=np.zeros(len(dmri_array_id_quickbundles))
            a[clusters[cluster].indices] = 1
            cluster_mask_quickbundles = (a.astype(bool)) & subject_mask_quickbundles

            subject_cluster_fmri_arrays_endpoint[subject_id][cluster] = fmri_array_endpoint[cluster_mask]
            subject_cluster_fmri_arrays_endpoint_quickbundles[subject_id][cluster] = fmri_array_endpoint_quickbundles[cluster_mask_quickbundles]
            subject_cluster_quickbundles_arrays[subject_id][cluster] = streamlines[cluster_mask_quickbundles]
 

    from dipy.io.streamline import load_tractogram, save_vtk_streamlines      
    for subject in range(len(subject_cluster_quickbundles_arrays)):
        os.makedirs(os.path.join(params['outputDirectory'],str(subject)),exist_ok=True)
        for cluster in range(len(subject_cluster_quickbundles_arrays[subject])):
            output_vtk_path = os.path.join(params['outputDirectory'], str(subject), f"{params['bundle']}_subject_{subject}_cluster_{cluster}_quickbundles.vtk")
            if len(subject_cluster_quickbundles_arrays[subject][cluster]) !=0:
                save_vtk_streamlines(subject_cluster_quickbundles_arrays[subject][cluster], output_vtk_path)
            
            


    for subject_id, cluster_data in subject_cluster_fmri_arrays_endpoint.items():
        subject_fmri_clusters_h5 = os.path.join(params['outputDirectory'], f"{params['bundle']}_subject_{subject_id}_fmri_clusters_endpoint.h5")
        with h5py.File(subject_fmri_clusters_h5, "w") as f:
            for cluster, cluster_fmri in cluster_data.items():
                f.create_dataset(f'cluster_{cluster}', data=cluster_fmri)
        utils.print_both(params['log'], f"\nSaved fMRI clusters for subject {subject_id} to:\t{subject_fmri_clusters_h5}")
    get_corr_origin(params['outputDirectory'],device)
    if params['baseline'] is False:
        for subject_id, cluster_data in subject_cluster_fmri_arrays_endpoint_quickbundles.items():
            subject_fmri_clusters_h5 = os.path.join(params['outputDirectory'], f"{params['bundle']}_subject_{subject_id}_fmri_clusters_endpoint_quickbundles.h5")
            with h5py.File(subject_fmri_clusters_h5, "w") as f:
                for cluster, cluster_fmri in cluster_data.items():
                    f.create_dataset(f'cluster_{cluster}', data=cluster_fmri)
            utils.print_both(params['log'], f"\nSaved fMRI clusters for subject {subject_id} to:\t{subject_fmri_clusters_h5}")
        get_corr_origin_quickbundles(params['outputDirectory'],device)

        
        from utils.clusters import compute_cluster_centroids
        # Convert clusters indices to array of cluster assignments
        cluster_assignments = np.zeros(len(fiber_array.fiber_array_ras), dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_assignments[cluster.indices] = cluster_idx

        cluster_centroids, cluster_reordered_fibers, cluster_alphas = compute_cluster_centroids(num_clusters, cluster_assignments, fiber_array.fiber_array_ras)
        utils.print_both(params['log'], f"alpha of quick bundle: { np.nanmean(cluster_alphas)}")
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



    

        