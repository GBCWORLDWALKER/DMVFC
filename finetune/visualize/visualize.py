from __future__ import print_function, division

import argparse
import re
import umap
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.cluster import KMeans
from cnslab_fmri.net.st_gcn import st_gcn_fiber
import training_functions_fiber_pair
from utils.calculate_index_pair import generate_unique_pairs

from utils import utils, nets
from utils.utils import str2bool, params_to_str
from utils.io import process_args_test, read_data, output_data
from utils.clusters import metrics_calculation, outlier_removal, metrics_subject_calculation
import utils.fibers as fibers
from mapping import MappingLossModel, calculate_mapping_loss, load_trained_net, train_mapping
import numpy as np
import os, h5py
from visualize_test import visualize_test

def kmeans_update(model,embeddings,num_clusters):
    km = KMeans(n_clusters=num_clusters, n_init=20)
    predicted=km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    if np.unique(predicted).shape[0] < num_clusters:
        print("kmeans:", np.unique(predicted))
        print("Error: empty clusters in kmeans")
        exit()

    weights = torch.from_numpy(centroids)
    if params['data_parallel']:
        model.module.clustering.set_weight(weights.to(params['device']))
    else:
        model.clustering.set_weight(weights.to(params['device']))
    return centroids
def visualize_clusters(model, model_b, model_mapping, dataset, dataset_b, params):
    preds = None
    probs = None
    probs_a = None
    probs_b = None
    subnum = None
    preds_a = None
    preds_b = None
    centroids_a = None
    centroids_b = None
    centroids_mapping_a = None
    centroids_mapping_b = None
    preds_mapping = None
    probs_mapping = None
    probs_mapping_a = None
    probs_mapping_b = None
    preds_mapping_b = None
    preds_mapping_a = None
    embeddings = None
    embeddings_b = None
    embeddings_mapping_a = None
    embeddings_mapping_b = None
    
    model.eval()
    model_b.eval()
    model_mapping.eval()

    dataset.get_epoch_similarity(0)
    dataset_b.get_epoch_similarity(0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
    dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=params["batch_size"], shuffle=False)
    for data, data_b in zip(dataloader, dataloader_b):
        
        input1, input2, _, _, subid = data
        input1_b, input2_b, _, _, subid_b = data_b
        
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])    
        input1_b = input1_b.to(params['device'])
        input2_b = input2_b.to(params['device'])    
        

        outputs_b, clusters_b, _, embedding_b, _, dis_point_b = model_b(input1_b, input2_b)
        outputs, clusters, _, embedding, _, dis_point = model(input1, input2)
        embedding_mapping_a, embedding_mapping_b = model_mapping(embedding, embedding_b)


        
        if embeddings is not None:

            subnum = np.concatenate((subnum, subid.cpu().detach().numpy().astype(int)), 0)
            embeddings = np.concatenate((embeddings, embedding.cpu().detach().numpy()), 0)
            embeddings_b = np.concatenate((embeddings_b, embedding_b.cpu().detach().numpy()), 0)
            embeddings_mapping_a = np.concatenate((embeddings_mapping_a, embedding_mapping_a.cpu().detach().numpy()), 0)
            embeddings_mapping_b = np.concatenate((embeddings_mapping_b, embedding_mapping_b.cpu().detach().numpy()), 0)
        else:

            subnum = subid.cpu().detach().numpy().astype(int)
            embeddings = embedding.cpu().detach().numpy()
            embeddings_b = embedding_b.cpu().detach().numpy()
            embeddings_mapping_a = embedding_mapping_a.cpu().detach().numpy()
            embeddings_mapping_b = embedding_mapping_b.cpu().detach().numpy()

        # kmeans_update(model,embeddings)
        # kmeans_update(model_b,embeddings_b)
    centroid_indices_mapping_a = []
    centroid_indices_mapping_b = []
    if params['mapping']:
        
        centroids_mapping_a=kmeans_update(model,embeddings_mapping_a,params['num_clusters'])
        centroids_mapping_b=kmeans_update(model_b,embeddings_mapping_b,params['num_clusters'])
    
        for centroid in centroids_mapping_a:
            distances = np.linalg.norm(embeddings_mapping_a - centroid, axis=1)
            centroid_index = np.argmin(distances)
            centroid_indices_mapping_a.append(centroid_index)

        for centroid in centroids_mapping_b:
            distances = np.linalg.norm(embeddings_mapping_b - centroid, axis=1)
            centroid_index = np.argmin(distances)
            centroid_indices_mapping_b.append(centroid_index)
        embeddings_mapping_a=torch.tensor(embeddings_mapping_a).to(params['device'])
        embeddings_mapping_b=torch.tensor(embeddings_mapping_b).to(params['device'])
        
        clusters,_=model.module.clustering(embeddings_mapping_a)
        clusters_b,_=model_b.module.clustering(embeddings_mapping_b)
        embeddings_mapping_b=embeddings_mapping_b.cpu().detach().numpy()
        embeddings_mapping_a=embeddings_mapping_a.cpu().detach().numpy()
        probs_mapping, preds_mapping = torch.max((clusters + clusters_b) / 2, 1)
        probs_mapping_a, preds_mapping_a = torch.max(clusters, 1)
        probs_mapping_b, preds_mapping_b = torch.max(clusters_b, 1)
        
    centroid_indices_a=[]
    centroid_indices_b=[]
    if params['none_mapping']:
        centroids_a=kmeans_update(model,embeddings,params['num_clusters'])
        centroids_b=kmeans_update(model_b,embeddings_b,params['num_clusters'])
        for centroid in centroids_a:
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            centroid_index = np.argmin(distances)
            centroid_indices_a.append(centroid_index)
        for centroid in centroids_b:
            distances = np.linalg.norm(embeddings_b - centroid, axis=1)
            centroid_index = np.argmin(distances)
            centroid_indices_b.append(centroid_index)
        embeddings=torch.tensor(embeddings).to(params['device'])
        embeddings_b=torch.tensor(embeddings_b).to(params['device'])
        clusters,_=model.module.clustering(embeddings)
        clusters_b,_=model_b.module.clustering(embeddings_b)
        embeddings_b=embeddings_b.cpu().detach().numpy()
        embeddings=embeddings.cpu().detach().numpy()
        probs_list=[]
        preds_list=[]
        for alpha in [1]:
            probs, preds = torch.max(alpha*clusters + (1-alpha)*clusters_b, 1)
            probs_list.append(probs.cpu().detach().numpy())
            preds_list.append(preds.cpu().detach().numpy())
        probs_a, preds_a = torch.max(clusters, 1)
        probs_b, preds_b = torch.max(clusters_b, 1)
        

        
        
        
        # centroids = []
        # for cluster_id in np.unique(preds):
        #     cluster_points = embeddings[preds == cluster_id]
        #     centroid = cluster_points.mean(axis=0)
        #     centroid_idx = np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))
        #     centroids.append(centroid_idx)
    # centroids_b= []
    # for cluster_id in np.unique(preds_b):
    #     cluster_points = embeddings_b[preds_b == cluster_id]
    #     centroid = cluster_points.mean(axis=0)
    #     centroid_idx = np.argmin(np.linalg.norm(cluster_points - centroid, axis=1))
    #     centroids_b.append(centroid_idx)

   
        

    return {
        'embeddings': embeddings,
        'embeddings_b': embeddings_b,
        'embeddings_mapping_a': embeddings_mapping_a,
        'embeddings_mapping_b': embeddings_mapping_b,
        'preds': preds.cpu().detach().numpy() if preds is not None else None,
        'probs': probs.cpu().detach().numpy() if probs is not None else None,
        'probs_a': probs_a.cpu().detach().numpy() if probs_a is not None else None,
        'probs_b': probs_b.cpu().detach().numpy() if probs_b is not None else None,
        'subnum': subnum,
        'preds_a': preds_a.cpu().detach().numpy() if preds_a is not None else None,
        'preds_b': preds_b.cpu().detach().numpy() if preds_b is not None else None,
        'centroids_a': centroid_indices_a,
        'centroids_b': centroid_indices_b,
        'centroids_mapping_a': centroid_indices_mapping_a,
        'centroids_mapping_b': centroid_indices_mapping_b,
        'preds_mapping': preds_mapping.cpu().detach().numpy() if preds_mapping is not None else None,
        'probs_mapping': probs_mapping.cpu().detach().numpy() if probs_mapping is not None else None,
        'probs_mapping_a': probs_mapping_a.cpu().detach().numpy() if probs_mapping_a is not None else None,
        'probs_mapping_b': probs_mapping_b.cpu().detach().numpy() if probs_mapping_b is not None else None,
        'preds_mapping_b': preds_mapping_b.cpu().detach().numpy() if preds_mapping_b is not None else None,
        "preds_mapping_a":preds_mapping_a.cpu().detach().numpy() if preds_mapping_a is not None else None,
        "probs_list":np.array(probs_list),
        "preds_list":np.array(preds_list)
        
    }
def visualize_clusters_b_only(model_b, dataset_b, params):
    preds_b = None
    probs_b = None
    subnum_b = None
    model_b.eval()
    if params['fmri_similarity_path'] != "None":
        dataset_b.get_epoch_similarity(0)
    else:
        dataset_b.get_epoch_similarity(0)
    dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=params["batch_size"], shuffle=False)
    for data_b in dataloader_b:
        
        input1_b, input2_b, _, _, subid_b = data_b
        
        input1_b = input1_b.to(params['device'])
        input2_b = input2_b.to(params['device'])    
        
        outputs_b, clusters_b, _, embedding_b, _, dis_point_b = model_b(input1_b, input2_b)

        probs_single_b, preds_single_b = torch.max(clusters_b, 1)
        
        if preds_b is not None:
            preds_b = np.concatenate((preds_b, preds_single_b.cpu().detach().numpy()), 0)
            probs_b = np.concatenate((probs_b, probs_single_b.cpu().detach().numpy()), 0)
            subnum_b = np.concatenate((subnum_b, subid_b.cpu().detach().numpy().astype(int)), 0)
            embeddings_b = np.concatenate((embeddings_b, embedding_b.cpu().detach().numpy()), 0)
        else:
            preds_b = preds_single_b.cpu().detach().numpy()
            probs_b = probs_single_b.cpu().detach().numpy()
            subnum_b = subid_b.cpu().detach().numpy().astype(int)
            embeddings_b = embedding_b.cpu().detach().numpy()

    return embeddings_b, preds_b, probs_b, subnum_b
def umap_visualize(embedding_2d, preds_final,probs,name,args):


    # Configure and fit UMAP
    if args['centroids']:
        centroids = args['centroids']
    else:
        centroids = None
    # Create scatter plot
    plt.figure(figsize=(15, 12))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1],
                         c=preds_final, cmap='tab20',
                         alpha=probs, s=5)
    if centroids is not None:
        plt.scatter(embedding_2d[centroids[0], 0], embedding_2d[centroids[0], 1],
                    c='yellow', marker='x', s=100, label='Centroids_a')
        plt.scatter(embedding_2d[centroids[1], 0], embedding_2d[centroids[1], 1],
                    c='green', marker='x', s=100, label='Centroids_b')
    plt.title('UMAP visualization of fiber embeddings', fontsize=14, pad=20)
    plt.xlabel('UMAP 1', fontsize=12, labelpad=10)
    plt.ylabel('UMAP 2', fontsize=12, labelpad=10)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    if centroids is not None:
        plt.legend()

    # Save plot with higher resolution
    os.makedirs(params['outputDirectory'], exist_ok=True) 
    output_path = os.path.join(params['outputDirectory'], f'umap_embeddings_{name}.png')

    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    utils.print_both(params['log'], "\nSaved UMAP visualization to: " + 
                    os.path.join(params['outputDirectory'], f'umap_embeddings_{name}.png'))
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
    parser.add_argument('--bundle', default=None, help='bundle name')
    parser.add_argument('--baseline', default=False, type=str2bool, help='baseline or not')
    parser.add_argument('--origin', default=False, type=str2bool, help='origin or not')
    parser.add_argument('--trained_net_mapping', default=None, help='path of trained net mapping')
    parser.add_argument('--data_parallel', default=False, type=str2bool, help='data parallel or not')
    parser.add_argument('--model_b', default="st", type=str, help='model b')
    parser.add_argument('--output_name', default="fmri_visualize", type=str, help='output name')
    parser.add_argument('--mapping', default=False, type=str2bool, help='mapping or not')
    parser.add_argument('--none_mapping', default=False, type=str2bool, help='none mapping or not')
    parser.add_argument('--presaved_path', default=None, type=str, help='presaved path')
    parser.add_argument('--num_clusters', default=7, type=int, help='num clusters')
    parser.add_argument('--not_pre_saved_results', default=False, type=str2bool, help='not pre saved results or not')
    parser.add_argument('--test', default=False, type=str2bool, help='test or not')
    parser.add_argument('--figure', default=False, type=str2bool, help='figure or not')
    parser.add_argument('--old_endpoint', default=False, type=str2bool, help='old endpoint or not')
    args = parser.parse_args()
    params = process_args_test(args)
    params['device'] = torch.device("cuda:1")
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
    
    index_pair=index_pair.unsqueeze(1).expand(-1, 80, -1, -1).flatten(start_dim=1,end_dim=-2).cpu().numpy()

    fmri_path_list = [os.path.join(params['multi_view_test'], subject) 
                      for subject in sorted(os.listdir(params['multi_view_test']), key=numsort) 
                      if params['bundle'] in subject]
    fmri_array=[]
    
    for fmri_path in fmri_path_list:
        f=sio.loadmat(fmri_path)
        fmri_array.append(f['srvf_values'])
    fmri_array=np.concatenate(fmri_array,axis=0)


    
    fmri_array_1=fmri_array
    if params['model_b']=="dgcnn":
        fmri_array_1=fmri_array[:,[0,-1],::2].reshape(-1,40,30)
        
    fmri_array_endpoint=fmri_array[:,:,::2]
    
    dataset_b=fibers.FiberPair(fmri_array_1,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]),
                               fmri=True,
                               model_b=params['model_b'])

    dataset = fibers.FiberPair(fiber_array.fiber_array_ras,
                               fiber_array.fiber_subID,
                               index_pair=index_pair,
                               subID_counts=subID_counts,
                               fmri_similarity_path=None,
                               dmri_similarity_path=None,
                               bundle=params['bundle'],
                               transform=transforms.Compose([transforms.ToTensor()]),
                               fmri=False,
                               model_b=params['model_b'])
    params['dataset_size'] = len(dataset)


    # GPU check
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    utils.print_both(params['log'], "\nPerforming calculations on:\t" + str(device))
    params['device'] = device
    weights_b = torch.load(params['trained_net_b'],weights_only=True)
    
    num_clusters = weights_b['module.clustering.weight'].size()[0]
    embedding_dimension = weights_b['module.clustering.weight'].size()[1]
    params['embedding_dimension']=embedding_dimension
    params['num_clusters']=num_clusters
    if params['model_b']=="st":
        model_b =  st_gcn_fiber(1, 1, embedding_dimension=embedding_dimension,num_clusters=num_clusters)
    elif params['model_b']=="dgcnn":
        idx_b = nets.get_dgcnn_idx(10, 40, params["batch_size"]).to(device)
        model_b = nets.DGCNN(k=10, input_channel=30, num_clusters=num_clusters, embedding_dimension=embedding_dimension, idx=idx_b,device=device)

    # embeddings_b, preds_b, probs_b, subnum_b = visualize_clusters_b_only(model_b, dataset_b, params)
    # # Create DGCNN model
    # weights = torch.load(params['trained_net'])
    # num_clusters = weights['clustering.weight'].size()[0]
    # embedding_dimension = weights['clustering.weight'].size()[1]
    
    params['mapping']=False
    params['none_mapping']=True
    idx = nets.get_dgcnn_idx(5, params["num_points"], params["batch_size"]).to(device)
    model = nets.DGCNN(k=5, input_channel=3, num_clusters=num_clusters, embedding_dimension=embedding_dimension, idx=idx,device=device)
    
    weights = torch.load(params['trained_net'],weights_only=True)
    weights_mapping=torch.load(params['trained_net_mapping'],weights_only=True)
    model_mapping=MappingLossModel(params)
    # weights_b = {'module.' + k: v for k, v in weights_b.items()}  

    # weights = {'module.' + k: v for k, v in weights.items()}  
    weights_mapping = {'module.' + k: v for k, v in weights_mapping.items()}  
    if torch.cuda.device_count() > 1:
        device_ids = [1,2]
        model = nn.DataParallel(model, device_ids=device_ids)
        model_b = nn.DataParallel(model_b, device_ids=device_ids)
        model_mapping=nn.DataParallel(model_mapping, device_ids=device_ids)
    model_b.load_state_dict(weights_b)
    model.load_state_dict(weights)
    model_mapping.load_state_dict(weights_mapping)
    model_b.to(device)
    model.to(device)
    model_mapping.to(device)

    

    # weights_b = torch.load(params['trained_net_b'])
    # num_clusters_b = weights_b['clustering.weight'].size()[0]
    # embedding_dimension_b = weights_b['clustering.weight'].size()[1]
    # idx_b = nets.get_dgcnn_idx(10, 40, params["batch_size"]).to(device)
    # model_b = nets.DGCNN(k=10, input_channel=30, num_clusters=num_clusters_b, embedding_dimension=embedding_dimension_b, idx=idx_b,device=device)
    # model_b = model_b.to(device)
    # model_b.load_state_dict(weights_b)

    dic_result = visualize_clusters(model,model_b, model_mapping,dataset,dataset_b, params)
    
    
    
    
    np.save(os.path.join(params['outputDirectory'], 'embeddings_mapping_a.npy'), dic_result['embeddings_mapping_a'])
    np.save(os.path.join(params['outputDirectory'], 'embeddings_mapping_b.npy'), dic_result['embeddings_mapping_b'])

    np.save(os.path.join(params['outputDirectory'], 'embeddings_a.npy'), dic_result['embeddings'])
    np.save(os.path.join(params['outputDirectory'], 'embeddings_b.npy'), dic_result['embeddings_b'])
    np.save(os.path.join(params['outputDirectory'], 'centroids_a.npy'), dic_result['centroids_a'])
    np.save(os.path.join(params['outputDirectory'], 'centroids_b.npy'), dic_result['centroids_b'])
    np.save(os.path.join(params['outputDirectory'], 'centroids_mapping_a.npy'), dic_result['centroids_mapping_a'])
    np.save(os.path.join(params['outputDirectory'], 'centroids_mapping_b.npy'), dic_result['centroids_mapping_b'])
    np.save(os.path.join(params['outputDirectory'], 'preds_mapping.npy'), dic_result['preds_mapping'])
    np.save(os.path.join(params['outputDirectory'], 'preds_mapping_a.npy'), dic_result['preds_mapping_a'])
    np.save(os.path.join(params['outputDirectory'], 'preds_mapping_b.npy'), dic_result['preds_mapping_b'])
    np.save(os.path.join(params['outputDirectory'], 'probs_mapping.npy'), dic_result['probs_mapping'])
    np.save(os.path.join(params['outputDirectory'], 'probs_mapping_a.npy'), dic_result['probs_mapping_a'])
    np.save(os.path.join(params['outputDirectory'], 'probs_mapping_b.npy'), dic_result['probs_mapping_b'])
    np.save(os.path.join(params['outputDirectory'], 'probs.npy'),dic_result['probs'])
    np.save(os.path.join(params['outputDirectory'], 'probs_a.npy'),dic_result['probs_a'])
    np.save(os.path.join(params['outputDirectory'], 'probs_b.npy'),dic_result['probs_b'])
    np.save(os.path.join(params['outputDirectory'], 'preds.npy'),dic_result['preds'])
    np.save(os.path.join(params['outputDirectory'], 'preds_a.npy'),dic_result['preds_a'])
    np.save(os.path.join(params['outputDirectory'], 'preds_b.npy'),dic_result['preds_b'])
    np.save(os.path.join(params['outputDirectory'], 'probs_list.npy'),dic_result['probs_list'])
    np.save(os.path.join(params['outputDirectory'], 'preds_list.npy'),dic_result['preds_list'])


    
    args = {'centroids':[dic_result['centroids_a'],dic_result['centroids_b']],'centroids_mapping':[dic_result['centroids_mapping_a'],dic_result['centroids_mapping_b']]}
    
    if params['none_mapping'] and params['figure']:
        reducer = umap.UMAP(random_state=42)
        embedding_2d = reducer.fit_transform(dic_result['embeddings'])
        embedding_2d_b = reducer.fit_transform(dic_result['embeddings_b'])
        umap_visualize(embedding_2d, dic_result['preds'],dic_result['probs'],"a_combined",args)
        umap_visualize(embedding_2d_b, dic_result["preds"],dic_result["probs"],"b_combined",args)
        umap_visualize(embedding_2d, dic_result["preds_a"],dic_result["probs_a"],"a_a",args)
        umap_visualize(embedding_2d_b, dic_result["preds_a"],dic_result["probs_a"],"b_a",args)
        umap_visualize(embedding_2d, dic_result["preds_b"],dic_result["probs_b"],"a_b",args)
        umap_visualize(embedding_2d_b, dic_result["preds_b"],dic_result["probs_b"],"b_b",args)

    if params['mapping'] and params['figure']:
        reducer1=umap.UMAP(random_state=42)
        embedding_2d_mapping_a = reducer1.fit_transform(dic_result['embeddings_mapping_a'])
        embedding_2d_mapping_b = reducer1.fit_transform(dic_result['embeddings_mapping_b'])
        umap_visualize(embedding_2d_mapping_a, dic_result["preds_mapping"],dic_result["probs_mapping"],"a_combined_mapping",args)
        umap_visualize(embedding_2d_mapping_b, dic_result["preds_mapping"],dic_result["probs_mapping"],"b_combined_mapping",args)
        umap_visualize(embedding_2d_mapping_a, dic_result["preds_mapping_a"],dic_result["probs_mapping_a"],"a_a_mapping",args)
        umap_visualize(embedding_2d_mapping_b, dic_result["preds_mapping_a"],dic_result["probs_mapping_a"],"b_a_mapping",args)
        umap_visualize(embedding_2d_mapping_a, dic_result["preds_mapping_b"],dic_result["probs_mapping_b"],"a_b_mapping",args)
        umap_visualize(embedding_2d_mapping_b, dic_result["preds_mapping_b"],dic_result["probs_mapping_b"],"b_b_mapping",args)
    if params['test']:
        params['not_pre_saved_results']=False
        params['presaved_path']= params['outputDirectory']
        params['fmri_path']=params['multi_view_test']
        visualize_test(params,fiber_array)