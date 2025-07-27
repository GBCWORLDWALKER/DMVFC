
import torch
import scipy.io as sio
import os
import pickle
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
def distance_prepare_single_tract(index_pair,subID_counts, fiber_array_ras, embedding_surf,epoch,filepath, bundle):
    from joblib import Parallel, delayed
    import numpy as np

    num_epochs=epoch+1
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    num_fibers=len(fiber_array_ras)
    def process_epoch(epoch):
        similarities = np.zeros(num_fibers)
        j=0
        sump=0

        for i in range(num_fibers):
            if i >=sump+subID_counts[j]:
                sump+=subID_counts[j]
                j+=1

            idx1, idx2 = index_pair[j][epoch, i-sump]
            fiber1 = fiber_array_ras[idx1+sump]
            fiber2 = fiber_array_ras[idx2+sump]
            similarity = distance_single(fiber1, fiber2, embedding_surf)
            similarities[i] = similarity
        
        # Save similarities for this epoch
        similarities_tensor = torch.tensor(similarities, device=device).to(torch.float32)
        torch.save(similarities_tensor,os.path.join(filepath,f'dmri_{bundle}_epoch_{epoch}.pt'))
    Parallel(n_jobs=32)(delayed(process_epoch)(epochs) for epochs in range(num_epochs))
    print("Distance preparation completed for all epochs.")
def distance_prepare_tract(index_pair,subID_counts, fiber_array_ras,fmri_path_list,epoch,filepath):
    from joblib import Parallel, delayed
    import numpy as np

    num_epochs=epoch+1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_fibers=len(fiber_array_ras)
    def process_epoch(epoch):
        similarities = np.zeros(num_fibers)
        j=0
        sump=0
        fmri_data = sio.loadmat(fmri_path_list[0])['Df']
        single_prepared = torch.load(os.path.join(filepath,f'single_tract_epoch_{epoch}.pt'))
        for i in range(num_fibers):
            if i >=sump+subID_counts[j]:
                sump+=subID_counts[j]
                j+=1
                fmri_data = sio.loadmat(fmri_path_list[j])['Df']
            idx1, idx2 = index_pair[j][epoch, i-sump]
            similarities[i] = single_prepared[i]+fmri_data[idx1,idx2]
            
        
        # Save similarities for this epoch
        similarities_tensor = torch.tensor(similarities, device=device).to(torch.float32)
        torch.save(similarities_tensor,os.path.join(filepath,f'similarities_epoch_{epoch}.pt'))
    Parallel(n_jobs=-1)(delayed(process_epoch)(epochs) for epochs in range(num_epochs))
    print("Distance preparation completed for all epochs.")
# def distance_prepare_dmri(filepath,index_pair,subID_counts, fiber_array_ras, embedding_surf,fmri_path_list,epoch,params):
#     from joblib import Parallel, delayed
#     import numpy as np

#     num_epochs = epoch + 1
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_fibers = len(fiber_array_ras)

#     def process_epoch(epoch):
#         similarities = torch.zeros(num_fibers)
#         j = 0
#         sump = 0
#         fmri_data = torch.tensor(sio.loadmat(fmri_path_list[epoch])['distances'],device=device,dtype=torch.float32)
#         single_prepared = torch.load(os.path.join(filepath, f'single_tract_epoch_{epoch}.pt')).to(device)
#         for i in range(num_fibers):
#             # idx1, idx2 = index_pair[j][epoch, i-sump]
#             # fiber1 = fiber_array_ras[idx1+sump]
#             # fiber2 = fiber_array_ras[idx2+sump]
#             # similarity = distance_single(fiber1, fiber2, embedding_surf)
#             similarities[i] = 0.6*single_prepared[i] + 0.4*fmri_data[j][i-sump]
#             if i >= sump + subID_counts[j] - 1:
#                 sump += subID_counts[j]
#                 j += 1
        
#         # Save similarities for this epoch immediately
#         similarities_tensor = similarities.to(torch.float32)
#         torch.save(similarities_tensor, os.path.join(filepath, f"similarities_epoch_{epoch}.pt"))
#         torch.cuda.empty_cache()
#         return epoch

#     results = Parallel(n_jobs=4)(delayed(process_epoch)(epochs) for epochs in range(num_epochs))

#     completed_epochs = [epochs for epochs in results if epochs is not None]
#     print(f"Distance preparation completed for epochs: {completed_epochs}")
def distance_prepare(similarity_path, bundle, fiber_array_ras, fmri_path,dmri_similarity_path, epoch, params):
    from joblib import Parallel, delayed
    import numpy as np
    num_epochs = epoch + 1
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise ValueError("No GPUs available. Please ensure at least one GPU is available.")
    os.makedirs(os.path.join(similarity_path),exist_ok=True)
    num_fibers = len(fiber_array_ras)

    def process_epoch(epoch, device_id):
        if os.path.exists(os.path.join(similarity_path, f"similarities_{bundle}_{epoch}.pt")):
            print(f"Similarities for epoch {epoch} already exist")
            return epoch
        device = torch.device(f"cuda:{device_id}")
        similarities = torch.zeros(num_fibers, device=device)
        # with open(os.path.join(fmri_path, f"fmri_{bundle}_{epoch}.pkl"), 'rb') as f:
        #     fmri_data = pickle.load(f)
        # fmri_data=torch.tensor(fmri_data,dtype=torch.float).reshape(-1).to(device)
        single_prepared_path = os.path.join(dmri_similarity_path, f'dmri_{bundle}_epoch_{epoch}.pt')
        single_prepared = torch.load(single_prepared_path, map_location=device,weights_only=True).to(device)
        similarities = (1-params['alpha'])*single_prepared # +params['alpha']*fmri_data
        similarities_tensor = similarities.to(torch.float32)
        save_path = os.path.join(similarity_path, f"similarities_{bundle}_{epoch}.pt")
        torch.save(similarities_tensor, save_path)
        torch.cuda.empty_cache()
        print(f"Similarities for epoch {epoch} finished")
        return epoch

    # Prepare list of (epoch, device_id) tuples
    epoch_device_pairs = [(ep, ep % num_gpus) for ep in range(num_epochs)]

    # Use Joblib's Parallel to process epochs in parallel across GPUs
    # results = Parallel(n_jobs=num_gpus)(
    #     delayed(process_epoch)(ep, device_id) for ep, device_id in epoch_device_pairs
    # )
    results=[process_epoch(ep, device_id) for ep, device_id in epoch_device_pairs]

    completed_epochs = [ep for ep in results if ep is not None]
    print(f"Distance preparation completed for epochs: {completed_epochs}")


    
def cluster_prepare(index_pair,subID_counts, fiber_array_ras, embedding_surf,fmri_path_list,epoch,params):
    num_epochs=epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_fibers=len(fiber_array_ras)
    for epoch in range(num_epochs):
        similarities = torch.zeros(num_fibers, device=device)
        j=0
        sump=0
        fmri_data = sio.loadmat(fmri_path_list[epoch+params['num_epochs']])['Df']
        for i in range(num_fibers):
            if i >=sump+subID_counts[j]:
                sump+=subID_counts[j]
                j+=1
                fmri_data = sio.loadmat(fmri_path_list[j])['Df']
            idx1, idx2 = index_pair[j][epoch, i-sump]
            fiber1 = fiber_array_ras[idx1+sump]
            fiber2 = fiber_array_ras[idx2+sump]
            similarity = distance_single(fiber1, fiber2, embedding_surf)
            similarities[i] = similarity+fmri_data[idx1,idx2]
        
        # Save similarities for this epoch
        torch.save(similarities, f'cluster_epoch_{epoch}.pt')

    print("Distance preparation completed for all epochs.")

