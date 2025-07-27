from utils import utils
import time
import torch
import numpy as np
import numpy
import copy
import os
import h5py
from sklearn.cluster import KMeans, AgglomerativeClustering
from utils.nets import ClusterlingLayer
from utils.distance_prepare import cluster_prepare  # Add this import at the top of the file
from utils.calculate_index_pair import generate_unique_pairs
from mapping import MappingLossModel, calculate_mapping_loss, load_trained_net, train_mapping
def test_model_multi(model, model_b, dataset,dataset_b, params):
    since = time.time()
    if params["surf"]:
        print("loading atlas pred info")
        with h5py.File(params['atlas_pred'], "r") as f:
            surf_cluster = f['surf_cluster'][:]
            surf_cluster = torch.tensor(surf_cluster).to(params['device'])
    else:
        print("no atlas pred")
        surf_cluster = None

    preds, probs, _ = calculate_predictions_roi_multi(model, model_b, copy.deepcopy(dataset), copy.deepcopy(dataset_b), params, surf_cluster=surf_cluster)
    time_elapsed = time.time() - since
    utils.print_both(params['log'], 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return preds, probs
def test_model(model, dataset, params):

    # Note the time
    since = time.time()

    if params["surf"]:
        print("loading atlas pred info")
        with h5py.File(params['atlas_pred'], "r") as f:
            surf_cluster = f['surf_cluster'][:]
            surf_cluster = torch.tensor(surf_cluster).to(params['device'])
    else:
        print("no atlas pred")
        surf_cluster = None

    preds, probs, _ = calculate_predictions_roi(model, copy.deepcopy(dataset), params, surf_cluster=surf_cluster)
    time_elapsed = time.time() - since
    utils.print_both(params['log'], 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return preds, probs

# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model_a, model_b, dataset_a,dataset_b, criteria, optimizers_a, optimizers_b, schedulers_a, schedulers_b, params, fiber_array,writer):

    # Note the time
    since = time.time()

 
    if writer is not None: board = True
    log = params['log']
    pretrained_net_a = params['model_files'][0]
    pretrained_net_b = params['model_files'][1]
    pretrain_net_path_a = params['model_files'][2]
    pretrain_net_path_b = params['model_files'][3]
    final_net_path_a = params['model_files'][4]
    final_net_path_b = params['model_files'][5]
    mapping_net_path = params['model_files'][6]
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch_size']
    num_epochs = params['epochs']
    pretrain_epochs = params['epochs_pretrain']
    gamma = params['gamma']


    update_interval = params['update_interval']

    # Pretrain or load weights
    if pretrain:

        utils.print_both(log, '\nPretraining...')

        if False:
            pretrained_net = "/data01/fan/SWM-DFC/data/SWM_f20k_n1065/EmbedDisSurf_LossDisSurf/models/DGCNN_003_pretrain_k2000/model_ep00215_l3.704068.pt"
            pretrained_weights = torch.load(pretrained_net)
            model.load_state_dict(pretrained_weights)

        pretrained_model_a,pretrained_model_b = pretraining(model_a, model_b, copy.deepcopy(dataset_a),copy.deepcopy(dataset_b), criteria[0], optimizers_a[1], optimizers_b[1], schedulers_a[1], schedulers_b[1], pretrain_epochs, params, writer)

        model_a = pretrained_model_a
        model_b = pretrained_model_b
        torch.save(model_a.state_dict(), pretrain_net_path_a)
        torch.save(model_b.state_dict(), pretrain_net_path_b)

        utils.print_both(log, '\nInitializing cluster centers using K-means')
        # model, predicted_km = Agglomerative(model, copy.deepcopy(dataset), params)
        # kmeans_initial(model_a, copy.deepcopy(dataset_a),model_b, params) # Original problematic line
        
        utils.print_both(log, 'Initializing cluster centers for model_a using K-means')
        model_a, _, _ = kmeans_initial(model_a, copy.deepcopy(dataset_a), params)

        utils.print_both(log, 'Initializing cluster centers for model_b using K-means')
        model_b, _, _ = kmeans_initial(model_b, copy.deepcopy(dataset_b), params)


        utils.print_both(log, '\nSaving pretrained model to:'+ pretrain_net_path_a)
        torch.save(model_a.state_dict(), pretrain_net_path_a)
      
        utils.print_both(log, '\nSaving pretrained model to:'+ pretrain_net_path_b)
        torch.save(model_b.state_dict(), pretrain_net_path_b)

    else:
        # try:
        print("device count:",torch.cuda.device_count())
        print("device name:",torch.cuda.get_device_name(0))  # For the first CUDA device
        pretrained_weights_a = torch.load(pretrained_net_a,map_location=device,weights_only=True)
        pretrained_weights_b = torch.load(pretrained_net_b,map_location=device,weights_only=True)
        if params['data_parallel']:
            pretrained_weights_a = {'module.'+ k: v for k,v in pretrained_weights_a.items()}
            pretrained_weights_b = {'module.'+ k: v for k,v in pretrained_weights_b.items()}

        reclustering = params["reclustering"]
        if not reclustering:
            utils.print_both(log, 'Pretrained weights loaded from file: ' + str(pretrained_net_a))
            utils.print_both(log, 'Pretrained weights loaded from file: ' + str(pretrained_net_b))
            model_a.load_state_dict(pretrained_weights_a)
            model_b.load_state_dict(pretrained_weights_b)
            # predicted_km = model.clustering.weight
        else:
            if params['data_parallel']:
                prev_num_clusters = pretrained_weights_a['module.clustering.weight'].size()[0]
            else:
                prev_num_clusters = pretrained_weights_a['clustering.weight'].size()[0]
            new_num_clusters  = params['num_clusters']
            if params['data_parallel']:
                embed_dim_a = pretrained_weights_a['module.clustering.weight'].size()[1]
            else:
                embed_dim_a = pretrained_weights_a['clustering.weight'].size()[1]
            if params['data_parallel']:
                embed_dim_b = pretrained_weights_b['module.clustering.weight'].size()[1]
            else:
                embed_dim_b = pretrained_weights_b['clustering.weight'].size()[1]
            utils.print_both(log, '\nRe-clustering with k from %d to %d' % (prev_num_clusters, new_num_clusters))

            for model in [model_a, model_b]:
                model.num_clusters = prev_num_clusters
                if model == model_a:
                    weights = torch.zeros((prev_num_clusters, embed_dim_a))
                else:
                    weights = torch.zeros((prev_num_clusters, embed_dim_b))
                if params['data_parallel']:
                    model.module.clustering.set_weight(weights.to(params['device']))
                else:
                    model.clustering.set_weight(weights.to(params['device']))
            print("pretrained_weights_a")
            model_a.load_state_dict(pretrained_weights_a)
            print("pretrained_weights_b")
            model_b.load_state_dict(pretrained_weights_b)

            for model in [model_a, model_b]:
                if params['data_parallel']:
                    model.module.num_clusters = new_num_clusters
                else:
                    model.num_clusters = new_num_clusters
                if model == model_a:
                    if params['data_parallel']:
                        model.module.clustering = ClusterlingLayer(embed_dim_a, new_num_clusters).to(params['device'])
                    else:
                        model.clustering = ClusterlingLayer(embed_dim_a, new_num_clusters).to(params['device'])
                else:
                    if params['data_parallel']:
                        model.module.clustering = ClusterlingLayer(embed_dim_b, new_num_clusters).to(params['device'])
                    else:
                        model.clustering = ClusterlingLayer(embed_dim_b, new_num_clusters).to(params['device'])
            if params['strategy_initial'] == 1 or params['strategy_initial'] == 0:
                kmeans_initial(model_b, copy.deepcopy(dataset_b),params)
            elif params['strategy_initial'] == 2:
                model_a, _,centers_indices = kmeans_initial(model_a, copy.deepcopy(dataset_a),params)
                model_b = initial_model_b(model_b,copy.deepcopy(dataset_b),centers_indices,params)
            else:
                model_a, predicted_km_a = kmeans(model_a, copy.deepcopy(dataset_a),params)
                model_b, predicted_km_b = kmeans(model_b, copy.deepcopy(dataset_b),params)
            utils.print_both(log, '\nSaving pretrained models to:' + pretrain_net_path_a)
            torch.save(model_a.state_dict(), pretrain_net_path_a)
            utils.print_both(log, '\nSaving pretrained models to:' + pretrain_net_path_b)
            torch.save(model_b.state_dict(), pretrain_net_path_b)

        # except Exception as e:
        #     print("Error, when loading pretrained weights")
        #     print(e)
        #     exit()



    #---------------------------------------------------------------------------------------------------
    # Mapping
    if params['mapping']:
        torch.cuda.empty_cache()
        epochs_mapping = params['epochs_mapping']
        model=train_mapping(model_a, model_b, dataset_a, dataset_b, params, epochs_mapping,writer,mapping_net_path)
        torch.save(model.state_dict(), mapping_net_path)
        utils.print_both(log, '\nMapping model saved to:' + mapping_net_path)
    #---------------------------------------------------------------------------------------------------
    
    
    
    
    
    utils.print_both(log, '\nCompute cluster surf from initial clustering, if loss_surf is used.')
    surf_flag = params["loss_surf"]
    surf_cluster_a=None
    surf_cluster_b=None

    utils.print_both(log, '\nInitialize target distribution')
    preds_initial_a, probs_initial_a, subID_per_fiber_a = calculate_predictions_roi(model_a, copy.deepcopy(dataset_a), params, surf_cluster=surf_cluster_a)
    preds_initial_b, probs_initial_b, subID_per_fiber_b = calculate_predictions_roi(model_b, copy.deepcopy(dataset_b), params, surf_cluster=surf_cluster_b)

    utils.print_both(log, '\nBegin clustering training')
    preds_uptated_a = torch.tensor(preds_initial_a).to(device)
    preds_uptated_b = torch.tensor(preds_initial_b).to(device)
    if params['demo']:
        num_epochs = 1
    for epoch in range(num_epochs):

        dataset_a.get_epoch_similarity(epoch+params['epochs_pretrain'])
        dataset_b.get_epoch_similarity(epoch+params['epochs_pretrain'])
        dataloader_a = torch.utils.data.DataLoader(dataset_a, batch_size=params["batch_size"], shuffle=False)
        dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=params["batch_size"], shuffle=False)
        utils.print_both(log, '\nEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(log,  '-' * 10)
        model_a.train(True)
        model_b.train(True)
        running_loss_a = 0.0
        running_loss_rec_a = 0.0
        running_loss_clust_a = 0.0
        running_loss_b = 0.0
        running_loss_rec_b = 0.0
        running_loss_clust_b = 0.0
        for batch_num, (data_a,data_b) in enumerate(zip(dataloader_a,dataloader_b)):
            # Uptade target distribution, check and print performance
            if batch_num % update_interval == 0 and not (batch_num == 0 and epoch == 0):
                utils.print_both(log, 'Updating cluster distribution.')


            # Get the inputs and labels
            input1, input2, dmri_similarity, index, _ = data_a
            input1 = input1.to(device)
            input2 = input2.to(device)
            dmri_similarity = dmri_similarity.to(device)
            index = index.to(device)
            input1_b, input2_b, fmri_similarity, index, _ = data_b
            input1_b = input1_b.to(device)
            input2_b = input2_b.to(device)
            fmri_similarity = fmri_similarity.to(device)
            index = index.to(device)
            # surf_bat = surf_bat.to(device)

            optimizers_a[0].zero_grad()
            optimizers_b[0].zero_grad()
            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs_a, clusters_a, _, _, _, dis_point_a = model_a(input1, input2, freeze=params['freeze'])
                outputs_b, clusters_b, _, _, _, dis_point_b = model_b(input1_b, input2_b)
                if surf_flag:
                    clusters_a = update_cluster_with_surf(surf_cluster_a, surf_cluster, dis_point_a, params)
                    clusters_b = update_cluster_with_surf(surf_cluster_b, surf_cluster, dis_point_b, params)

                tar_dist_a = target_distribution(clusters_a)
                tar_dist_b = target_distribution(clusters_b)
                loss_rec_a = criteria[0](outputs_a, dmri_similarity)
                loss_rec_b = criteria[0](outputs_b, fmri_similarity)
                if params['loss_strategy']==0:
                    loss_clust_a = gamma * criteria[1](torch.log(clusters_a), tar_dist_a)+gamma * criteria[1](torch.log(clusters_a), tar_dist_b)
                    loss_clust_b = gamma * criteria[1](torch.log(clusters_b), tar_dist_b)+gamma * criteria[1](torch.log(clusters_b), tar_dist_a)
                if params['loss_strategy']==1:
                    loss_clust_a = gamma * criteria[1](torch.log(clusters_a), tar_dist_b)
                    loss_clust_b = gamma * criteria[1](torch.log(clusters_b), tar_dist_a)
                if params['loss_strategy']==2:
                    if epoch % 2 == 0:
                        loss_clust_a=  criteria[1](torch.log(clusters_a), tar_dist_a)
                        loss_clust_b=  criteria[1](torch.log(clusters_b), tar_dist_a)
                    else:
                        loss_clust_a= criteria[1](torch.log(clusters_a), tar_dist_b)
                        loss_clust_b= criteria[1](torch.log(clusters_b), tar_dist_b)
                        #loss_clust_b=0.2*0.285*gamma * criteria[1](torch.log(clusters_b), tar_dist_b)
                if params['loss_strategy']==3:
                    if epoch<params['epochs']/2:
                        loss_clust_a=gamma * criteria[1](torch.log(clusters_a), tar_dist_a)
                        loss_clust_b=gamma * criteria[1](torch.log(clusters_b), tar_dist_a)
                    else:
                        loss_clust_a=gamma * criteria[1](torch.log(clusters_a), tar_dist_b)
                        loss_clust_b=gamma * criteria[1](torch.log(clusters_b), tar_dist_b)
                        
                loss_a = loss_rec_a + loss_clust_a
                loss_b = loss_rec_b + loss_clust_b
                loss_a.backward(retain_graph=True)
                loss_b.backward(retain_graph=True)
                optimizers_a[0].step()
                optimizers_b[0].step()

            schedulers_a[0].step()
            schedulers_b[0].step()

            # For keeping statistics
            running_loss_a += loss_a.item() * input1.size(0)
            running_loss_b += loss_b.item() * input1.size(0)
            running_loss_rec_a += loss_rec_a.item() * input1.size(0)
            running_loss_rec_b += loss_rec_b.item() * input1.size(0)
            running_loss_clust_a += loss_clust_a.item() * input1.size(0)
            running_loss_clust_b += loss_clust_b.item() * input1.size(0)

            if numpy.isnan(running_loss_a):
                print("Error: loss should not have NaN")
                exit()
            if numpy.isnan(running_loss_b):
                print("Error: loss should not have NaN")
                exit()

            # Some current stats
            loss_batch_a = loss_a.item()
            loss_batch_rec_a = loss_rec_a.item()
            loss_batch_clust_a = loss_clust_a.item()
            loss_accum_a = running_loss_a / (batch_num * batch + input1.size(0))
            loss_accum_rec_a = running_loss_rec_a / (batch_num * batch + input1.size(0))
            loss_accum_clust_a = running_loss_clust_a / (batch_num * batch + input1.size(0))
            loss_batch_b = loss_b.item()
            loss_batch_rec_b = loss_rec_b.item()
            loss_batch_clust_b = loss_clust_b.item()
            loss_accum_b = running_loss_b / (batch_num * batch + input1.size(0))
            loss_accum_rec_b = running_loss_rec_b / (batch_num * batch + input1.size(0))
            loss_accum_clust_b = running_loss_clust_b / (batch_num * batch + input1.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(log, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss_a {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery_a {5:.4f} ({6:.4f})\t'
                                           'Loss_clustering_a {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num + 1,
                                                                                        len(dataloader_a),
                                                                                        loss_batch_a,
                                                                                        loss_accum_a, loss_batch_rec_a,
                                                                                        loss_accum_rec_a,
                                                                                        loss_batch_clust_a,
                                                                                        loss_accum_clust_a))
                utils.print_both(log, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss_b {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery_b {5:.4f} ({6:.4f})\t'
                                           'Loss_clustering_b {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num + 1,
                                                                                        len(dataloader_b),
                                                                                        loss_batch_b,
                                                                                        loss_accum_b, loss_batch_rec_b,
                                                                                        loss_accum_rec_b,
                                                                                        loss_batch_clust_b,
                                                                                        loss_accum_clust_b))
                if board:
                    niter = epoch * len(dataloader_a) + (batch_num + 1)
                    writer.add_scalar('/Loss_a', loss_accum_a, niter)
                    writer.add_scalar('/Loss_recovery_a', loss_accum_rec_a, niter)
                    writer.add_scalar('/Loss_clustering_a', loss_accum_clust_a, niter)
                    writer.add_scalar('/Loss_b', loss_accum_b, niter)
                    writer.add_scalar('/Loss_recovery_b', loss_accum_rec_b, niter)
                    writer.add_scalar('/Loss_clustering_b', loss_accum_clust_b, niter)
                    

        epoch_loss_a = running_loss_a / dataset_size
        epoch_loss_rec_a = running_loss_rec_a / dataset_size
        epoch_loss_clust_a = running_loss_clust_a / dataset_size
        epoch_loss_b = running_loss_b / dataset_size
        epoch_loss_rec_b = running_loss_rec_b / dataset_size
        epoch_loss_clust_b = running_loss_clust_b / dataset_size

        if board:
            writer.add_scalar('/Loss_a' + '/Epoch', epoch_loss_a    , epoch + 1)
            writer.add_scalar('/Loss_rec_a' + '/Epoch', epoch_loss_rec_a, epoch + 1)
            writer.add_scalar('/Loss_clust_a' + '/Epoch', epoch_loss_clust_a, epoch + 1)
            writer.add_scalar('/Loss_b' + '/Epoch', epoch_loss_b, epoch + 1)
            writer.add_scalar('/Loss_rec_b' + '/Epoch', epoch_loss_rec_b, epoch + 1)
            writer.add_scalar('/Loss_clust_b' + '/Epoch', epoch_loss_clust_b, epoch + 1)

        utils.print_both(log, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss_a,epoch_loss_rec_a,epoch_loss_clust_a))
        utils.print_both(log, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss_b,epoch_loss_rec_b,epoch_loss_clust_b))

        # update subnum
        subnum_cluster_a = subnum_update_torch(model_a.num_clusters, preds_uptated_a, torch.tensor(subID_per_fiber_a).to(device), device)


        # result check
        preds_uptated_np = preds_uptated_a.cpu().detach().numpy()
        values, counts = numpy.unique(preds_uptated_np, return_counts=True)
        arg = numpy.argsort(counts)
        arg = numpy.flip(arg)
        values = values[arg]
        counts = counts[arg]
        numb_of_nonempty_clusters = numpy.unique(values).shape[0]
        utils.print_both(log, " * Check 1: Max and min number of fibers per cluster: %d - %d" % (counts.max(),counts.min()))
        utils.print_both(log, " * Check 2: Empty clusters: %d / %d are found." % (numb_of_nonempty_clusters, model_a.num_clusters))
        utils.print_both(log, " * Check 3: Max and min subject number per cluster:: %d - %d" % (subnum_cluster_a.max(), subnum_cluster_a.min()))
        # model_a,_=kmeans(model_b, copy.deepcopy(dataset_b),model_a, params)
        # model_b,_=kmeans(model_b, copy.deepcopy(dataset_b), model_b, params)
        if epoch%3==0:
            torch.save(model_a.state_dict(), os.path.join(params['outputDirectory'], 'model_a_clustering_epoch_'+str(epoch+1)+'.pt'))
            torch.save(model_b.state_dict(), os.path.join(params['outputDirectory'], 'model_b_clustering_epoch_'+str(epoch+1)+'.pt'))
    time_elapsed = time.time() - since
    utils.print_both(log, 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    if num_epochs == 0:
        preds = preds_initial_a
        probs = probs_initial_a
    else:
        preds, probs, _ = calculate_predictions_roi(model_a, dataset_a, params, surf_cluster=surf_cluster_a)

    if surf_flag:
        surf_cluster = surf_cluster_uptate_torch(model_a.num_clusters, preds, x_surf_a, device)
    else:
        surf_cluster = None

    return model_a, model_b, preds, probs, surf_cluster

# Pretraining function for recovery loss only

def pretraining(model_a, model_b, dataset_a,dataset_b, criterion, optimizer_a, optimizer_b, scheduler_a, scheduler_b, num_epochs, params,writer):
    # Note the time
    since = time.time()

    if writer is not None: board = True
    txt_file = params['log']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch_size']

    # Prep variables for weights and accuracy of the best model
    best_model_wts_a = copy.deepcopy(model_a.state_dict())
    best_model_wts_b = copy.deepcopy(model_b.state_dict())
    best_loss_a = best_loss_b = 10000.0
    since_epoch = time.time()   
    # Go through all epochs
    batch_num_all = 0
    for epoch in range(0, num_epochs):

        dataset_a.get_epoch_similarity(epoch + params['epochs_pretrain'])
        dataset_b.get_epoch_similarity(epoch + params['epochs_pretrain'])
        dataloader_a = torch.utils.data.DataLoader(dataset_a, batch_size=params["batch_size"], shuffle=False)
        dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=params["batch_size"], shuffle=False)
                
        # Split dataset into training and validation sets
        train_size_a = int(0.9 * len(dataset_a))
        val_size_a = len(dataset_a) - train_size_a
        train_dataset_a, val_dataset_a = torch.utils.data.random_split(dataset_a, [train_size_a, val_size_a])

        train_size_b = int(0.9 * len(dataset_b))
        val_size_b = len(dataset_b) - train_size_b
        train_dataset_b, val_dataset_b = torch.utils.data.random_split(dataset_b, [train_size_b, val_size_b])

        train_dataloader_a = torch.utils.data.DataLoader(train_dataset_a, batch_size=params["batch_size"], shuffle=False)
        val_dataloader_a = torch.utils.data.DataLoader(val_dataset_a, batch_size=params["batch_size"], shuffle=False)

        train_dataloader_b = torch.utils.data.DataLoader(train_dataset_b, batch_size=params["batch_size"], shuffle=False)
        val_dataloader_b = torch.utils.data.DataLoader(val_dataset_b, batch_size=params["batch_size"], shuffle=False)

        time_epoch = time.time() - since_epoch
        print('time_epoch:', time_epoch)
        utils.print_both(txt_file, 'time_epoch: {}'.format(time_epoch))
        since_epoch = time.time()
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)
        model_b.train(True)  # Set model to training mode
        model_a.train(True)  # Set model to training mode

        running_loss_a = running_loss_b = 0.0
        cur_lr_a = optimizer_a.param_groups[-1]['lr']
        cur_lr_b = optimizer_b.param_groups[-1]['lr']
        print('cur_lr_a:', cur_lr_a, 'cur_lr_b:', cur_lr_b)

        # Keep the batch number for inter-phase statistics(batch_size,15,3)
        batch_num = 1
        
        for data_a, data_b in zip(train_dataloader_a, train_dataloader_b):
            # Get the inputs and labels
            input1, input2, dmri_similarity, _, _ = data_a
            fmri_array1, fmri_array2, fmri_similarity, _, _ = data_b

            input1 = input1.to(device)
            input2 = input2.to(device)
            fmri_array1 = fmri_array1.to(device)
            fmri_array2 = fmri_array2.to(device)
            dmri_similarity = dmri_similarity.to(device)
            fmri_similarity = fmri_similarity.to(device)

            # zero the parameter gradients
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()

            with torch.set_grad_enabled(True):
                outputs_a, _, _, _, _, _ = model_a(input1, input2)
                outputs_b, _, _, _, _, _ = model_b(fmri_array1, fmri_array2)
                loss_a = criterion(outputs_a, dmri_similarity)*1e4
                loss_b = criterion(outputs_b, fmri_similarity)*1e4
                loss_a.backward()
                loss_b.backward()
                optimizer_a.step()
                optimizer_b.step()

            # For keeping statistics
            running_loss_a += loss_a.item() * input1.size(0)
            running_loss_b += loss_b.item() * fmri_array1.size(0)

            # Some current stats
            loss_batch_a = loss_a.item()
            loss_batch_b = loss_b.item()
            loss_accum_a = running_loss_a / ((batch_num - 1) * batch + input1.size(0))
            loss_accum_b = running_loss_b / ((batch_num - 1) * batch + fmri_array1.size(0))

            print_freq = 1000
            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss_a {3:.4f} ({4:.4f})\tLoss_b {5:.4f} ({6:.4f})\t'.format(
                               epoch + 1, batch_num, len(train_dataloader_a),
                               loss_batch_a, loss_accum_a,
                               loss_batch_b, loss_accum_b))
                if writer:
                    niter = epoch * len(train_dataloader_a) + batch_num
                    writer.add_scalar('Pretraining/Loss_a', loss_accum_a, niter)
                    writer.add_scalar('Pretraining/Loss_b', loss_accum_b, niter)

            batch_num = batch_num + 1
            batch_num_all = batch_num_all + 1
                # Validation phase
        model_a.eval()
        model_b.eval()
        val_running_loss_a = 0.0
        val_running_loss_b = 0.0
        with torch.no_grad():
            for data_a, data_b in zip(val_dataloader_a, val_dataloader_b):
                input1_a, input2_a, sim_score_a, _, _ = data_a
                input1_b, input2_b, sim_score_b, _, _ = data_b

                input1_a = input1_a.to(device)
                input2_a = input2_a.to(device)
                sim_score_a = sim_score_a.to(device)
                input1_b = input1_b.to(device)
                input2_b = input2_b.to(device)
                sim_score_b = sim_score_b.to(device)

                outputs_a, _, _, _, _, _ = model_a(input1_a, input2_a)
                outputs_b, _, _, _, _, _ = model_b(input1_b, input2_b)
                loss_a = criterion(outputs_a, sim_score_a)*1e4
                loss_b = criterion(outputs_b, sim_score_b)*1e4

                val_running_loss_a += loss_a.item() * input1_a.size(0)
                val_running_loss_b += loss_b.item() * input1_b.size(0)

        val_loss_a = val_running_loss_a / val_size_a
        val_loss_b = val_running_loss_b / val_size_b
        utils.print_both(txt_file, 'Validation Loss_a: {:.4f}'.format(val_loss_a))
        utils.print_both(txt_file, 'Validation Loss_b: {:.4f}'.format(val_loss_b))

        scheduler_a.step()
        scheduler_b.step()

        epoch_loss_a = running_loss_a / train_size_a
        epoch_loss_b = running_loss_b / train_size_b

        if epoch == 0: 
            first_loss_a = epoch_loss_a
            first_loss_b = epoch_loss_b

        if epoch == 4 and (epoch_loss_a / first_loss_a > 1 or epoch_loss_b / first_loss_b > 1):
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if writer:
            writer.add_scalars('Pretraining/Loss_a', {'Train': epoch_loss_a, 'Validation': val_loss_a}, epoch + 1)
            writer.add_scalars('Pretraining/Loss_b', {'Train': epoch_loss_b, 'Validation': val_loss_b}, epoch + 1)

        utils.print_both(txt_file, 'Pretraining:\t Loss_a: {:.4f}\tLoss_b: {:.4f}'.format(epoch_loss_a, epoch_loss_b))

        if epoch % 5 == 0:
            pretrain_net_path_a = params['model_files'][1] + '_a'
            pretrain_net_path_b = params['model_files'][1] + '_b'

            pretrain_net_epoch_folder_a = pretrain_net_path_a.replace('.pt', '')
            pretrain_net_epoch_folder_b = pretrain_net_path_b.replace('.pt', '')
            os.makedirs(pretrain_net_epoch_folder_a, exist_ok=True)
            os.makedirs(pretrain_net_epoch_folder_b, exist_ok=True)
            pretrain_net_epoch_path_a = os.path.join(pretrain_net_epoch_folder_a, "model_a_ep%05d_l%06f.pt" % (epoch, epoch_loss_a))
            pretrain_net_epoch_path_b = os.path.join(pretrain_net_epoch_folder_b, "model_b_ep%05d_l%06f.pt" % (epoch, epoch_loss_b))
            torch.save(model_a.state_dict(), pretrain_net_epoch_path_a)
            torch.save(model_b.state_dict(), pretrain_net_epoch_path_b)

        # If wanted to add some criterium in the future
        if epoch_loss_a < best_loss_a:
            best_loss_a = epoch_loss_a
            best_model_wts_a = copy.deepcopy(model_a.state_dict())
        if epoch_loss_b < best_loss_b:
            best_loss_b = epoch_loss_b
            best_model_wts_b = copy.deepcopy(model_b.state_dict())

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model_a.load_state_dict(best_model_wts_a)
    model_b.load_state_dict(best_model_wts_b)
    model_a.pretrained = True
    model_b.pretrained = True

    return model_a, model_b

# K-means clusters initialisation
def initial_model_b(model_b, dataset_b,centroids_indices, params):
    dataset_b.get_epoch_similarity(params['epochs_pretrain'])
    dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=int(params["batch_size"]/2), shuffle=False)
    embedding_array = None
    model_b.eval()
    for data in dataloader_b:
        input1, input2,  _, _, _ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        _, _, _, embedding, _, _ = model_b(input1, input2)

        if embedding_array is not None:
            embedding_array = np.concatenate((embedding_array, embedding.cpu().detach().numpy()), 0)
        else:
            embedding_array = embedding.cpu().detach().numpy()
    if params['data_parallel']:
        model_b.module.clustering.set_weight(torch.from_numpy(embedding_array[centroids_indices]).to(params['device']))
    else:
        model_b.clustering.set_weight(torch.from_numpy(embedding_array[centroids_indices]).to(params['device']))
    return model_b
def kmeans_initial(model, dataset, params):

    dataset.get_epoch_similarity(params['epochs_pretrain'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(params["batch_size"]/2), shuffle=False)

    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    
    model.eval()

    embedding_array = None
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        input1, input2,  _, _, _ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        _, _, _, embedding, _, _ = model(input1, input2)

        if embedding_array is not None:
            embedding_array = np.concatenate((embedding_array, embedding.cpu().detach().numpy()), 0)
        else:
            embedding_array = embedding.cpu().detach().numpy()

    # Perform K-means
    predicted = km.fit_predict(embedding_array)
    
    if numpy.unique(predicted).shape[0] < model.num_clusters:
        print("kmeans:", numpy.unique(predicted))
        print("Error: empty clusters in kmeans")
        exit()

    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    centers_indices = []
    for center in km.cluster_centers_:
        distances = np.linalg.norm(embedding_array - center, axis=1)
        center_index = np.argmin(distances)
        centers_indices.append(center_index)
    if params['data_parallel']:
        model.module.clustering.set_weight(copy.deepcopy(weights).to(params['device']))
    else:
        model.clustering.set_weight(copy.deepcopy(weights).to(params['device']))

    return model, predicted, centers_indices
def kmeans(model, dataset,  params):

    dataset.get_epoch_similarity(params['epochs_pretrain'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(params["batch_size"]/2), shuffle=False)

    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    
    model.eval()

    embedding_array = None
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        input1, input2,  _, _, _ = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        _, _, _, embedding, _, _ = model(input1, input2)

        if embedding_array is not None:
            embedding_array = np.concatenate((embedding_array, embedding.cpu().detach().numpy()), 0)
        else:
            embedding_array = embedding.cpu().detach().numpy()

    # Perform K-means
    predicted = km.fit_predict(embedding_array)
    
    if numpy.unique(predicted).shape[0] < model.num_clusters if not params['data_parallel'] else model.module.num_clusters:
        print("kmeans:", numpy.unique(predicted))
        print("Error: empty clusters in kmeans")
        exit()

    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    if params['data_parallel']:
        model.module.clustering.set_weight(weights.to(params['device']))
    else:
        model.clustering.set_weight(weights.to(params['device']))

    return model, predicted



def Agglomerative(model,dataset, params):

    model.eval()
    dataset.get_epoch_similarity(params['epochs_pretrain'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(params["batch_size"]/2), shuffle=False)
    embedding_array = None
    # surf_array = None
    subid_array = None
    print('put all data together')
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        input1, input2, _, _, subid = data
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])
        _, _, _, embedding, _, _ = model(input1, input2)

        if embedding_array is not None:
            embedding_array = np.concatenate((embedding_array, embedding.cpu().detach().numpy()), 0)
            # surf_array = np.concatenate((surf_array, surf1.cpu().detach().numpy()), 0)
            subid_array = np.concatenate((subid_array, subid.cpu().detach().numpy()), 0)
        else:
            embedding_array = embedding.cpu().detach().numpy()
            subid_array = subid.cpu().detach().numpy()

    # downsample to save memory
    embedding_array = embedding_array[::params['clustering_fiber_interval'], :]
    # surf_array = surf_array[::params['clustering_fiber_interval'], :]
    subid_array = subid_array[::params['clustering_fiber_interval']]

    print("embedding_array shape", embedding_array.shape)

    # uni_rows ,_ ,_ = np.unique(surf_array, return_counts=True, return_index=True, axis=0)

    # bat_size = 10000
    # if surf_array.shape[0] < bat_size: # for testing
    #     uni_idx, uni_ind = np.where((uni_rows[:, None, :] == surf_array).all(2))
    # else:
    #     num_batch = int(surf_array.shape[0] / bat_size)
    #     uni_idx = []
    #     uni_ind = []
    #     for b_idx in range(num_batch):
    #         bat_start = b_idx * bat_size
    #         bat_end = (b_idx + 1) * bat_size

    #         surf_array_bat = surf_array[bat_start:bat_end, :]
    #         uni_idx_, uni_ind_ = np.where((uni_rows[:, None, :] == surf_array_bat).all(2))

    #         uni_idx.append(uni_idx_)
    #         uni_ind.append(uni_ind_ + bat_start)

    #     uni_idx = numpy.concatenate(uni_idx)
    #     uni_ind = numpy.concatenate(uni_ind)

    #     tmp_sort = numpy.argsort(uni_ind)
    #     uni_idx = uni_idx[tmp_sort]
    #     uni_ind = uni_ind[tmp_sort]

    avg_nof_per_cluster = subid_array.shape[0] / model.num_clusters

    # subTH = 0.3  # initial clusters with very low number of subjects will not be considered at this stage.
    noc_list = []
    fibers_with_same_regions_list = []
    # for idx in np.unique(uni_idx):
    #     print(idx, uni_idx.shape)
    #     fibers_with_same_regions = uni_ind[uni_idx == idx]
    #     subid_fibers_same_regions = subid_array[fibers_with_same_regions]

    #     wmpg_fibers_same_regions = np.unique(subid_fibers_same_regions).shape[0] / np.unique(subid_array).shape[0]
    #     if wmpg_fibers_same_regions < subTH:
    #         continue

    #     nof = fibers_with_same_regions.shape[0]
    #     if nof < avg_nof_per_cluster / 4:
    #         continue

    #     noc = numpy.round(nof / avg_nof_per_cluster)
    #     if noc == 0:
    #         noc = 1.0
    #     noc_list.append(int(noc))
    #     fibers_with_same_regions_list.append(fibers_with_same_regions)

    # noc_total_tmp = sum(noc_list)
    # diff_noc = noc_total_tmp - model.num_clusters
    # print("## input noc = %d, initial noc = %d " % (model.num_clusters, noc_total_tmp))
    # if diff_noc <= 0:
    #     sorted_noc_ind = np.argsort(noc_list)[::-1]
    #     noc_list = numpy.array(noc_list)
    #     noc_list[sorted_noc_ind[:-diff_noc]] += 1
    # else:
    #     sorted_noc_ind = np.argsort(noc_list)[::-1]
    #     noc_list = numpy.array(noc_list)
    #     noc_list[sorted_noc_ind[:diff_noc]] -= 1

    # print("total NoC", sum(noc_list))

    cluster_centers_ = None
    predicted = np.zeros((subid_array.shape[0])) - 1
    count_cluster = 0
    for noc, fibers_with_same_regions in zip(noc_list, fibers_with_same_regions_list):

        print(noc)
        embedding_array_ = embedding_array[fibers_with_same_regions, :]
        km = KMeans(n_clusters=noc, n_init=20)

        predicted_ = km.fit_predict(embedding_array_)

        predicted[fibers_with_same_regions] = predicted_ + count_cluster
        count_cluster += noc

        if cluster_centers_ is not None:
            cluster_centers_ = np.concatenate((cluster_centers_, km.cluster_centers_), 0)
        else:
            cluster_centers_ = km.cluster_centers_
    print(numpy.unique(predicted).shape[0])
    if numpy.unique(predicted).shape[0] < model.num_clusters:
        print("kmeans:", numpy.unique(predicted))
        print("Error: empty clusters in kmeans")
        exit()

    # Update clustering layer weights
    weights = torch.from_numpy(cluster_centers_)
    model.clustering.set_weight(weights.to(params['device']))

    return model, predicted
def calculate_predictions_roi_multi(model, model_b, dataset,dataset_b, params, surf_cluster=None):
    
    preds = None
    probs = None
    subnum = None
    model.eval()
    model_b.eval()

    dataset.get_epoch_similarity(0)
    dataset_b.get_epoch_similarity(0)   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
    dataloader_b = torch.utils.data.DataLoader(dataset_b, batch_size=params["batch_size"], shuffle=False)
    for data,data_b in zip(dataloader,dataloader_b):
        
        input1, input2, _, _, subid = data
        input1_b, input2_b, _, _, subid_b = data_b
        
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])    
        input1_b = input1_b.to(params['device'])
        input2_b = input2_b.to(params['device'])    
        
        outputs, clusters, _, _, _, dis_point = model(input1, input2)
        outputs_b, clusters_b, _, _, _, dis_point_b = model_b(input1_b, input2_b)

        if surf_cluster is not None:
            # if surf_bat.shape[1] > surf_cluster.shape[1]:
            #     surf_cluster = torch.cat([surf_cluster, torch.zeros(surf_cluster.shape[0], surf_bat.shape[1]-surf_cluster.shape[1])], dim=1)
            clusters = update_cluster_with_surf(surf_bat, surf_cluster, dis_point, params)
        if params['baseline'] is not True:
            probs_single, preds_single = torch.max((clusters+clusters_b)/2,1)
        else:
            probs_single, preds_single = torch.max(clusters,1)
        #probs_single, preds_single = torch.max(clusters,1)
        if preds is not None:
            preds = np.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = np.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
            subnum = np.concatenate((subnum, subid.cpu().detach().numpy().astype(int)), 0)
        else:
            preds = preds_single.cpu().detach().numpy()
            probs = probs_single.cpu().detach().numpy()
            subnum = subid.cpu().detach().numpy().astype(int)

    return preds, probs, subnum
def calculate_predictions_roi(model, dataset, params, surf_cluster=None):
    
    preds = None
    probs = None
    subnum = None
    model.eval()

    dataset.get_epoch_similarity(0)   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
    for data in dataloader:
        
        input1, input2, _, _, subid = data
        
        input1 = input1.to(params['device'])
        input2 = input2.to(params['device'])    

        
        outputs, clusters, _, _, _, dis_point = model(input1, input2)

        if surf_cluster is not None:
            # if surf_bat.shape[1] > surf_cluster.shape[1]:
            #     surf_cluster = torch.cat([surf_cluster, torch.zeros(surf_cluster.shape[0], surf_bat.shape[1]-surf_cluster.shape[1])], dim=1)
            clusters = update_cluster_with_surf(surf_bat, surf_cluster, dis_point, params)

        probs_single, preds_single = torch.max(clusters,1)
        
        if preds is not None:
            preds = np.concatenate((preds, preds_single.cpu().detach().numpy()), 0)
            probs = np.concatenate((probs, probs_single.cpu().detach().numpy()), 0)
            subnum = np.concatenate((subnum, subid.cpu().detach().numpy().astype(int)), 0)
        else:
            preds = preds_single.cpu().detach().numpy()
            probs = probs_single.cpu().detach().numpy()
            subnum = subid.cpu().detach().numpy().astype(int)

    return preds, probs, subnum


# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def surf_cluster_uptate_torch(num_clusters, preds, x_surf, device):
    surf_cluster = torch.zeros([num_clusters, x_surf.shape[1]])
    for i in range(num_clusters):
        t = x_surf[preds == i]
        t1 = torch.sum(t, 0)
        if t1.sum() == 0:
            continue
        t1 = t1 / t1.sum()
        v, ind = torch.topk(t1, 2, sorted=True, largest=True)
        t1_ = torch.zeros_like(t1)
        t1_[ind] = t1[ind]
        # if v[1] < 0.1:
        #     t1_[ind[0]] = 1
        # else:
        #     t1_[ind] = 0.5
        surf_cluster[i] = t1_
    surf_cluster = surf_cluster.to(device)
    return surf_cluster


def subnum_update_torch(num_clusters, preds, subnum, device):
    sub_cluster = torch.zeros(num_clusters)
    for i in range(num_clusters):
        t = subnum[preds == i]
        sub_cluster[i] = torch.unique(t).shape[0]
    sub_cluster = sub_cluster.to(device)
    return sub_cluster


def update_cluster_with_surf(surf_bat, surf_cluster, dis_point, params):

    tmp = torch.sum(surf_bat, dim=1)
    surf_bat[tmp == 1, :] *= 2  # fiber connecting to the same region
    surf_bat /= 2  # change value to [1, 0] and [0.5, 0.5]

    surf_sim = []
    for c_idx in range(surf_cluster.shape[0]):
        c_surf = surf_cluster[c_idx, :]
        c_diff = surf_bat - c_surf
        c_diff = c_diff * c_diff
        c_l2 = torch.sum(c_diff, dim=1)
        surf_sim.append(c_l2)
    surf_sim = torch.stack(surf_sim).T

    dis_surf = surf_sim  # 1 - torch.mm(surf_bat.float(), surf_cluster.t().float())
    dis_surf[dis_surf < 0] = 0
    dis_surf = dis_surf.to(params['device'])

    x = 1.0 + dis_point * torch.pow(1 + dis_surf, 5)
    x = 1.0 / x
    x = torch.t(x) / torch.sum(x, dim=1)
    x = torch.t(x)
    clusters = x

    return clusters

def check_and_remove_empty_cluster(params, model, preds_uptated, dataloader_entire):

    cluster_sizes = numpy.array([torch.sum(preds_uptated == idc).item() for idc in range(model.num_clusters)])

    if numpy.any(cluster_sizes == 0):

        cluster_sorted_indices = numpy.argsort(cluster_sizes)
        cluster_sizes_sorted = cluster_sizes[cluster_sorted_indices]
        empty_indices = cluster_sorted_indices[cluster_sizes_sorted == 0]
        if empty_indices.shape[0] > model.num_clusters / 2:
            utils.print_both(params["log"], "ERROR: Too many empty clusters. Seem nothing we can do to help. Try a smaller K.")
            exit()

        model.eval()
        dataloader = copy.deepcopy(dataloader_entire)
        embedding_array = None
        for data in dataloader:
            input1, input2, _, _, _, _ = data
            input1 = input1.to(params['device'])
            input2 = input2.to(params['device'])
            _, _, _, embedding, _, _ = model(input1, input2)

            if embedding_array is not None:
                embedding_array = np.concatenate((embedding_array, embedding.cpu().detach().numpy()), 0)
            else:
                embedding_array = embedding.cpu().detach().numpy()

        for e_idx, empty_cluster in enumerate(empty_indices):
            split_cluster = cluster_sorted_indices[-(e_idx + 1)]

            utils.print_both(params["log"], " * Remove empty cluster %d (%d fibers), and split cluster %d (%d fibers)" %
                             (empty_cluster, cluster_sizes[empty_cluster], split_cluster, cluster_sizes[split_cluster]))

            fibers_in_split_cluster = torch.where(preds_uptated == split_cluster)[0].detach().cpu().numpy()

            embedding_array_split_cluster = embedding_array[fibers_in_split_cluster, :]

            km = KMeans(n_clusters=2)
            km_predicted = km.fit_predict(embedding_array_split_cluster)

            km_predicted_relabel = numpy.zeros_like(km_predicted)
            km_predicted_relabel[km_predicted == 0] = split_cluster
            km_predicted_relabel[km_predicted == 1] = empty_cluster
            km_predicted_relabel = torch.tensor(km_predicted_relabel).to(params['device']).type(preds_uptated.type())

            preds_uptated[fibers_in_split_cluster] = km_predicted_relabel

            km_weights = km.cluster_centers_

            model_cluster_weights = copy.deepcopy(model.clustering.weight).detach().cpu().numpy()
            model_cluster_weights[split_cluster, :] = km_weights[0, :]
            model_cluster_weights[empty_cluster, :] = km_weights[1, :]

            model_cluster_weights = torch.from_numpy(model_cluster_weights).to(params['device'])
            model.clustering.set_weight(model_cluster_weights)

    return model