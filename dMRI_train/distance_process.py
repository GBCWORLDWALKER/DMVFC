import torch
import os
from utils.fibers import fiber_pair_similarity

def distance_single(fiber1, fiber2, embedding_surf):
    if not embedding_surf:
        similarity = fiber_pair_similarity(fiber1, fiber2)
    else:
        similarity = fiber_pair_similarity(fiber1, fiber2)
    
    fiber1 = torch.tensor(fiber1.T, dtype=torch.float)
    fiber2 = torch.tensor(fiber2.T, dtype=torch.float)
    similarity = torch.tensor(similarity, dtype=torch.float)

    return similarity

def distance_prepare_single_tract(index_pair, subID_counts, fiber_array_ras, embedding_surf, epoch, filepath, bundle):
    from joblib import Parallel, delayed
    import numpy as np

    num_epochs=epoch+1 # epoch here is the maximum epoch index (0-based)
    # 移除device定义，不使用GPU
    num_fibers=len(fiber_array_ras)
    def process_epoch(current_epoch_idx): # Renamed 'epoch' to 'current_epoch_idx' to avoid confusion
        similarities = np.zeros(num_fibers)
        j=0
        sump=0

        for i in range(num_fibers):
            if i >=sump+subID_counts[j]:
                sump+=subID_counts[j]
                j+=1

            # Ensure index_pair access is correct for the structure: index_pair[subject_index][epoch, fiber_pair_index_in_subject]
            # Assuming index_pair is a list of tensors/arrays, one per subject.
            # And subID_counts is a dictionary mapping subject ID (or index j) to fiber count for that subject.
            idx1, idx2 = index_pair[j][current_epoch_idx, i-sump]
            fiber1 = fiber_array_ras[idx1+sump]
            fiber2 = fiber_array_ras[idx2+sump]
            similarity = distance_single(fiber1, fiber2, embedding_surf)
            similarities[i] = similarity
        
        # 修改以使用CPU处理数据
        similarities_tensor = torch.tensor(similarities).to(torch.float32)  # 不指定device参数
        torch.save(similarities_tensor, os.path.join(filepath, f'dmri_{bundle}_epoch_{current_epoch_idx}.pt'))
    
    Parallel(n_jobs=-1)(delayed(process_epoch)(epochs_idx) for epochs_idx in range(num_epochs)) # epochs_idx will go from 0 to epoch
    print(f"Distance preparation completed for bundle {bundle} for all epochs.")

if __name__ == "__main__":
    import argparse
    import pickle
    import os 
    import torch 
    from utils.io import process_args, read_data
    from utils.utils import str2bool, print_both

    parser = argparse.ArgumentParser(description='Prepare dMRI similarity matrices based on fiber pairs')
    parser.add_argument('-indir', action="store", dest="inputDirectory", default=None, help='A folder of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument('-outdir', action="store", dest="outputDirectory", default=None, help='Output folder (used for index_pair.pkl location and logs).')
    parser.add_argument('--embedding_surf', default=False, type=str2bool, help='Incorporating cortical information.')
    parser.add_argument('-fmri_path', action="store", dest="fmri_path", type=str, default=None, help='path to fmri data')
    parser.add_argument('--epochs', default=1, type=int, help='Clustering epochs (used in epoch calculation).')
    parser.add_argument('--epochs_pretrain', default=300, type=int, help='Pretraining epochs (used in epoch calculation).')
    parser.add_argument('--dmri_similarity_path', default=None, type=str, help='Path to save dMRI similarity .pt files (this will be the filepath argument).')
    parser.add_argument('-trf', action="store", dest="numberOfFibers_train", type=int, default=None, help='Number of fibers of each training data to analyze from each subject. None is all.')
    parser.add_argument('-l', action="store", dest="fiberLength", type=int, default=0, help='Minimum length (in mm) of fibers to analyze. 0mm is default.')
    parser.add_argument('-p', action="store", dest="numberOfFiberPoints", type=int, default=14, help='Number of points in each fiber to process. 14 is default.')
    parser.add_argument('--bundle_list_path', type=str, required=True, help='Path to the directory containing files whose names (without extension) will be used as bundle names.')
    parser.add_argument('--index_path', type=str, default=None, help='Path to the index_pair.pkl file. If not provided, will use the one in the input directory.')
    parser.add_argument('-bold',action="store", dest="bold", type=str2bool, default=True, help='path to bold data')
    parser.add_argument('--reclustering', default=True, type=str2bool, help='reclustering')
    parser.add_argument('--loss_subID', default=False, type=str2bool, help='include sub ID in loss computation')
    parser.add_argument('--loss_surf', default=False, type=str2bool, help='include surf in loss computation')
    parser.add_argument('--ro', default=True, type=str2bool, help='outlier removal')
    parser.add_argument('--full_brain', default=False, type=str2bool, help='full brain')
    parser.add_argument('--ro_std', default=1, type=int, help='outlier removal threshold')
    parser.add_argument('--num_clusters', default=2000, type=int, help='number of clusters')
    parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
    parser.add_argument('--clustering_fiber_interval', default=1, type=int, help='downsample of fibers for clustering')
    parser.add_argument('--output_clusters', default=True, type=str2bool, help='output cluster vtk files')
    parser.add_argument('--freeze', default=False, type=str2bool, help='freeze embedding training during clustering')
    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
    parser.add_argument('--pretrain', default=True, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--pretrained_net', default=None, help='index or path of pretrained net')
    parser.add_argument('--alpha', default=0, type=float, help='alpha for similarity')
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
    parser.add_argument('--similarity_path',default=None,type=str,help='similarity path')
    parser.add_argument('--fmri_similarity_path',default=None,type=str,help='fmri similarity path')
    parser.add_argument('--output_name',default=None,type=str,help='output name')
    # args.bundle will be used by process_args initially, but overwritten in the loop for read_data
    parser.add_argument('--bundle', type=str, default=None, help='Current bundle being processed (set internally).')


    args = parser.parse_args()
    # params will hold initial settings; params['bundle'] will be updated in the loop
    params = process_args(args) 

    print_both(params['log'], "===== dMRI Similarity Preparation Script =====")

    # Global parameters and data that are not bundle-specific
    epoch_param_for_func = params['epochs_pretrain'] + params['epochs'] * 2 
    print_both(params['log'], f"Max epoch index to process: {epoch_param_for_func}")

    index_pair_file_path = params['index_path']
    print_both(params['log'], f"Loading global index_pair from {index_pair_file_path}")
    if not os.path.exists(index_pair_file_path):
        print_both(params['log'], f"ERROR: index_pair.pkl not found at {index_pair_file_path}")
        exit(1)
    with open(index_pair_file_path, 'rb') as f:
        index_pair_data = pickle.load(f) # Assuming index_pair is global
    
    embedding_surf_val = params['embedding_surf']
    
    dmri_output_path = params.get('dmri_similarity_path') 
    if dmri_output_path is None:
        dmri_output_path = os.path.join(params['outputDirectory'], 'dmri_similarity_files')
        print_both(params['log'], f"dmri_similarity_path not provided, defaulting to: {dmri_output_path}")
    os.makedirs(dmri_output_path, exist_ok=True)
    print_both(params['log'], f"Output path for .pt files: {dmri_output_path}")

    # Get bundle_list from the directory specified by the command-line argument
    bundle_list_input_path = args.bundle_list_path
    if not os.path.isdir(bundle_list_input_path):
        print_both(params['log'], f"Error: bundle_list_path '{bundle_list_input_path}' does not exist or is not a directory.")
        exit(1)
    
    bundle_list = [os.path.splitext(f)[0] for f in os.listdir(bundle_list_input_path) if os.path.isfile(os.path.join(bundle_list_input_path, f))]
    print_both(params['log'], f"Found bundles to process: {bundle_list} in {bundle_list_input_path}")

    if not bundle_list:
        print_both(params['log'], "No bundles found in the specified bundle_list_path. Exiting.")
        exit(0)

    # Iterate through each bundle and perform dMRI similarity calculation
    for bundle_name in bundle_list:
        print_both(params['log'], f"Processing bundle: {bundle_name}")
        
        # Set the current bundle in params for read_data
        params['bundle'] = bundle_name 
        
        # Data preparation (now bundle-specific, inside the loop)
        print_both(params['log'], f"Reading data for bundle '{bundle_name}' from {params['inputDirectory']}")
        # read_data uses params['bundle'] to load bundle-specific files
        _, fiber_array_bundle = read_data(params, verbose=False) 
        
        if fiber_array_bundle is None or len(fiber_array_bundle.fiber_array_ras) == 0 :
            print_both(params['log'], f"Warning: No fibers found for bundle '{bundle_name}'. Skipping.")
            continue

        fiber_array_ras_bundle_data = fiber_array_bundle.fiber_array_ras
        
        # Get subID_counts for the current bundle
        unique_subIDs_bundle, counts_bundle = torch.unique(torch.tensor(fiber_array_bundle.fiber_subID), return_counts=True)
        subID_counts_bundle_data = dict(zip(unique_subIDs_bundle.tolist(), counts_bundle.tolist()))
        print_both(params['log'], f"SubID counts for bundle '{bundle_name}': {subID_counts_bundle_data}")

        if not subID_counts_bundle_data:
             print_both(params['log'], f"Warning: No subID counts for bundle '{bundle_name}'. Skipping.")
             continue
        
        distance_prepare_single_tract(
            index_pair=index_pair_data, # Global index_pair
            subID_counts=subID_counts_bundle_data, # Bundle-specific subID counts
            fiber_array_ras=fiber_array_ras_bundle_data, # Bundle-specific fiber data
            embedding_surf=embedding_surf_val,
            epoch=epoch_param_for_func, 
            filepath=dmri_output_path,   
            bundle=bundle_name # Current bundle being processed
        )
    
    print_both(params['log'], "All bundles processed.")
    if 'log' in params and hasattr(params['log'], 'close'):
        params['log'].close()