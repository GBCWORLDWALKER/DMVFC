from __future__ import print_function, division
import sys
import re
import os, h5py
from matplotlib.widgets import EllipseSelector
import torch
from torchvision import transforms
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
from cnslab_fmri.net.st_gcn import st_gcn_fiber
from corr_origin_quickbundles import get_corr as get_corr_origin_quickbundles
from corr_origin import get_corr as get_corr_origin
sys.path.append('..')
def visualize_test(params,fiber_array):

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
                      if params['bundle'] in subject] 
    # fmri_path_list = [os.path.join(params['multi_view_test'], subject,fmri_path) 
    #             for subject in sorted(os.listdir(params['multi_view_test']), key=numsort) for fmri_path in sorted(os.listdir(os.path.join(params['multi_view_test'],subject))) 
    #             if params['bundle'] in fmri_path]
    fmri_array=[]
    
    for fmri_path in fmri_path_list:
        f=sio.loadmat(fmri_path)
        fmri_array.append(f['srvf_values'])
    # for fmri_path in fmri_path_list:
    #     mat=sio.loadmat(fmri_path)
    #     fmri_array.append(mat['tsrvf_values'].transpose(2,1,0))
    fmri_array=np.concatenate(fmri_array,axis=0)
    if params['old_endpoint']:
        endpoint_path="/data06/jinwang/isbi/data/bundle_full/endpoint_test"
        fmri_path_list_endpoint = [os.path.join(endpoint_path, subject,fmri_path) 
                        for subject in sorted(os.listdir(endpoint_path), key=numsort) for fmri_path in sorted(os.listdir(os.path.join(endpoint_path,subject))) 
                        if params['bundle'] in fmri_path]
        fmri_array_endpoint=[]
        for fmri_path in fmri_path_list_endpoint:
            mat=sio.loadmat(fmri_path)
            fmri_array_endpoint.append(mat['W_all'].transpose(2,1,0))
        fmri_array_endpoint=np.concatenate(fmri_array_endpoint,axis=0)
    else:
        fmri_array_endpoint=fmri_array[:,[0,-1],:]



    #fmri_array_endpoint=fmri_array[:,[0,-1],:]
    fmri_array_1=fmri_array
    dataset_b=fibers.FiberPair(fmri_array_1,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]))

    dataset = fibers.FiberPair(fiber_array.fiber_array_ras,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]))
    params['dataset_size'] = len(dataset)




    # GPU check
    device = torch.device("cuda:"+str(params['gpu']) if torch.cuda.is_available() else "cpu")
    utils.print_both(params['log'], "\nPerforming calculations on:\t" + str(device))
    params['device'] = device
    preds_final_list=[]
    probs_final_list=[]
    if params['not_pre_saved_results']:
    # Create DGCNN model
        weights = torch.load(params['trained_net'])
        num_clusters = weights['clustering.weight'].size()[0]
        embedding_dimension = weights['clustering.weight'].size()[1]


        
        model_b =  st_gcn_fiber(1, 1, embedding_dimension=embedding_dimension,num_clusters=params['num_clusters'])
        model_b = model_b.to(device)
        model_b.load_state_dict(weights)
        idx = nets.get_dgcnn_idx(5, params["num_points"], params["batch_size"]).to(device)
        model = nets.DGCNN(k=5, input_channel=3, num_clusters=num_clusters, embedding_dimension=embedding_dimension, idx=idx,device=device)
        model = model.to(device)
        model.load_state_dict(weights)
        preds_final, probs_final =  training_functions_fiber_pair.test_model_multi(model,model_b, dataset,dataset_b, params)
    else:
        num_clusters=params['num_clusters']
        preds_final_list=np.load(os.path.join(params['presaved_path'], f'preds_list.npy'),allow_pickle=True)
        probs_final_list=np.load(os.path.join(params['presaved_path'], f'probs_list.npy'),allow_pickle=True)
    
    for i in range(len(preds_final_list)):
        params['alpha']=0.1*(i+1)
        preds_final=preds_final_list[i]
        probs_final=probs_final_list[i]
        # output result
        print("\n=======================")
        utils.print_both(params["log"], "\n# Outlier removal:")
        params["num_clusters"] = num_clusters
        rejected_fibers = outlier_removal(params, preds_final, probs_final)
        
        fmri_array_endpoint_quickbundles = fmri_array_endpoint
        fmri_array_endpoint_1 = np.delete(fmri_array_endpoint, rejected_fibers, axis=0)

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

                subject_cluster_fmri_arrays_endpoint[subject_id][cluster] = fmri_array_endpoint_1[cluster_mask]
                subject_cluster_fmri_arrays_endpoint_quickbundles[subject_id][cluster] = fmri_array_endpoint_quickbundles[cluster_mask_quickbundles]
                subject_cluster_quickbundles_arrays[subject_id][cluster] = streamlines[cluster_mask_quickbundles]
    
        if i==0:
            from dipy.io.streamline import load_tractogram, save_vtk_streamlines      
            for subject in range(len(subject_cluster_quickbundles_arrays)):
                os.makedirs(os.path.join(params['outputDirectory'],str(subject)),exist_ok=True)
                for cluster in range(len(subject_cluster_quickbundles_arrays[subject])):
                    output_vtk_path = os.path.join(params['outputDirectory'], str(subject), f"{params['bundle']}_subject_{subject}_cluster_{cluster}_quickbundles.vtk")
                    if len(subject_cluster_quickbundles_arrays[subject][cluster]) !=0:
                        save_vtk_streamlines(subject_cluster_quickbundles_arrays[subject][cluster], output_vtk_path)
                    
                


        for subject_id, cluster_data in subject_cluster_fmri_arrays_endpoint.items():
            os.makedirs(os.path.join(params['outputDirectory'],str((i+1)*0.1)),exist_ok=True)
            subject_fmri_clusters_h5 = os.path.join(params['outputDirectory'], str((i+1)*0.1),f"{params['bundle']}_subject_{subject_id}_fmri_clusters_endpoint.h5")
            with h5py.File(subject_fmri_clusters_h5, "w") as f:
                for cluster, cluster_fmri in cluster_data.items():
                    f.create_dataset(f'cluster_{cluster}', data=cluster_fmri)
            utils.print_both(params['log'], f"\nSaved fMRI clusters for subject {subject_id} to:\t{subject_fmri_clusters_h5}")
        get_corr_origin(os.path.join(params['outputDirectory'],str((i+1)*0.1)),device)
        if params['baseline'] is False:
            if i==0:
                for subject_id, cluster_data in subject_cluster_fmri_arrays_endpoint_quickbundles.items():
                    subject_fmri_clusters_h5 = os.path.join(params['outputDirectory'], str((i+1)*0.1),f"{params['bundle']}_subject_{subject_id}_fmri_clusters_endpoint_quickbundles.h5")
                    with h5py.File(subject_fmri_clusters_h5, "w") as f:
                        for cluster, cluster_fmri in cluster_data.items():
                            f.create_dataset(f'cluster_{cluster}', data=cluster_fmri)
                    utils.print_both(params['log'], f"\nSaved fMRI clusters for subject {subject_id} to:\t{subject_fmri_clusters_h5}")
                get_corr_origin_quickbundles(os.path.join(params['outputDirectory'],str((i+1)*0.1)),device)

            
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