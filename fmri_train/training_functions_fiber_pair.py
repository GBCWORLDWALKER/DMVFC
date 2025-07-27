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
def train_model(model,dataset , criteria, optimizers, schedulers, params, fiber_array,writer):

    # Note the time
    since = time.time()

    # Unpack parameters
    if writer is not None: board = True
    log = params['log']
    pretrained_net = params['model_files'][1]
    pretrain_net_path = params['model_files'][3]
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

        
        # pretrained_net = "/data06/jinwang/isbi/data/bundle_full/out_full_train/models/fmri_original_single_CC_5_pretrain_k5.pt"
        # pretrained_weights = torch.load(pretrained_net)
        # model.load_state_dict(pretrained_weights,strict=False)

        pretrained_model = pretraining(model, copy.deepcopy(dataset), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params,writer)

        model = pretrained_model
        torch.save(model.state_dict(), pretrain_net_path)

        utils.print_both(log, '\nInitializing cluster centers using K-means')
        # model, predicted_km = Agglomerative(model, copy.deepcopy(dataset), params)
        model, predicted_km = kmeans(model, copy.deepcopy(dataset), params)

        utils.print_both(log, '\nSaving pretrained model to:'+ pretrain_net_path)
        torch.save(model.state_dict(), pretrain_net_path)
        exit()

    else:
        try:
            pretrained_weights = torch.load(pretrained_net)

            reclustering = params["reclustering"]
            if not reclustering:
                utils.print_both(log, 'Pretrained weights loaded from file: ' + str(pretrained_net))
                model.load_state_dict(pretrained_weights)
                # predicted_km = model.clustering.weight
            else:
                prev_num_clusters = pretrained_weights['clustering.weight'].size()[0]
                new_num_clusters  = params['num_clusters']
                embed_dim = model.clustering.weight.size()[1]

                utils.print_both(log, '\nRe-clustering with k from %d to %d' % (prev_num_clusters, new_num_clusters))

                model.num_clusters = prev_num_clusters
                weights = torch.zeros((prev_num_clusters, embed_dim))
                model.clustering.set_weight(weights.to(params['device']))
                model.load_state_dict(pretrained_weights)

                model.num_clusters = new_num_clusters
                model.clustering = ClusterlingLayer(embed_dim, new_num_clusters).to(params['device'])
                #model, predicted_km = Agglomerative(model, dataset, params)
                model, predicted_km = kmeans(model, copy.deepcopy(dataset), params)

                utils.print_both(log, '\nSaving pretrained model to:' + pretrain_net_path)
                torch.save(model.state_dict(), pretrain_net_path)

        except Exception as e:
            print("Error, when loading pretrained weights")
            print(e)
            exit()

    utils.print_both(log, '\nCompute cluster surf from initial clustering, if loss_surf is used.')
    surf_flag = params["loss_surf"]
    if surf_flag:
        preds_km = torch.tensor(predicted_km).to(device)
        x_surf = torch.tensor(fiber_array.fiber_array_endpoints_onehot).to(device)
        x_surf_ds = x_surf[::params['clustering_fiber_interval'], :]
        surf_cluster = surf_cluster_uptate_torch(model.num_clusters, preds_km, x_surf_ds, device)
    else:
        surf_cluster = None

    utils.print_both(log, '\nInitialize target distribution')
    preds_initial, probs_initial, subID_per_fiber = calculate_predictions_roi(model, copy.deepcopy(dataset), params, surf_cluster=surf_cluster)
    utils.print_both(log, '\nBegin clustering training')
    preds_uptated = torch.tensor(preds_initial).to(device)
    for epoch in range(num_epochs):
        dataset.get_epoch_similarity(0)
        dataset.get_epoch_similarity(epoch+params['epochs_pretrain'])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)
        utils.print_both(log, '\nEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(log,  '-' * 10)

        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0
        for batch_num, data in enumerate(dataloader):
            # Uptade target distribution, check and print performance
            if batch_num % update_interval == 0 and not (batch_num == 0 and epoch == 0):
                utils.print_both(log, 'Updating cluster distribution.')
                if surf_flag:
                    surf_cluster = surf_cluster_uptate_torch(model.num_clusters, preds_uptated, x_surf, device)

            # Get the inputs and labels
            input1, input2, sim, index, _ = data
            input1 = input1.to(device)
            input2 = input2.to(device)
            sim = sim.to(device)
            index = index.to(device)
            # surf_bat = surf_bat.to(device)

            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _, _, _, dis_point = model(input1, input2)
                if surf_flag:
                    clusters = update_cluster_with_surf(surf_bat, surf_cluster, dis_point, params)

                _, preds = torch.max(clusters, 1)

                preds_uptated[index] = preds
                tar_dist = target_distribution(clusters)

                loss_rec = criteria[0](outputs, sim)
                loss_clust = gamma * criteria[1](torch.log(clusters), tar_dist)
                loss = loss_rec + loss_clust
                loss.backward()
                optimizers[0].step()

            schedulers[0].step()

            # For keeping statistics
            running_loss += loss.item() * input1.size(0)
            running_loss_rec += loss_rec.item() * input1.size(0)
            running_loss_clust += loss_clust.item() * input1.size(0)

            if numpy.isnan(running_loss):
                print("Error: loss should not have NaN")
                exit()

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / (batch_num * batch + input1.size(0))
            loss_accum_rec = running_loss_rec / (batch_num * batch + input1.size(0))
            loss_accum_clust = running_loss_clust / (batch_num * batch + input1.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(log, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss_clustering {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num + 1,
                                                                                        len(dataloader),
                                                                                        loss_batch,
                                                                                        loss_accum, loss_batch_rec,
                                                                                        loss_accum_rec,
                                                                                        loss_batch_clust,
                                                                                        loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + (batch_num + 1)
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        utils.print_both(log, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,epoch_loss_rec,epoch_loss_clust))

        # update subnum
        subnum_cluster = subnum_update_torch(model.num_clusters, preds_uptated, torch.tensor(subID_per_fiber).to(device), device)

        # result check
        preds_uptated_np = preds_uptated.cpu().detach().numpy()
        values, counts = numpy.unique(preds_uptated_np, return_counts=True)
        arg = numpy.argsort(counts)
        arg = numpy.flip(arg)
        values = values[arg]
        counts = counts[arg]
        numb_of_nonempty_clusters = numpy.unique(values).shape[0]
        utils.print_both(log, " * Check 1: Max and min number of fibers per cluster: %d - %d" % (counts.max(),counts.min()))
        utils.print_both(log, " * Check 2: Empty clusters: %d / %d are found." % (numb_of_nonempty_clusters, model.num_clusters))
        utils.print_both(log, " * Check 3: Max and min subject number per cluster:: %d - %d" % (subnum_cluster.max(), subnum_cluster.min()))

    time_elapsed = time.time() - since
    utils.print_both(log, 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    if num_epochs == 0:
        preds = preds_initial
        probs = probs_initial
    else:
        preds, probs, _ = calculate_predictions_roi(model, dataset, params, surf_cluster=surf_cluster)

    if surf_flag:
        surf_cluster = surf_cluster_uptate_torch(model.num_clusters, preds, x_surf, device)
    else:
        surf_cluster = None

    return model, preds, probs, surf_cluster

# Pretraining function for recovery loss only
def pretraining(model, dataset, criterion, optimizer, scheduler, num_epochs, params,writer):
    # Note the time
    since = time.time()


    if writer is not None: board = True
    txt_file = params['log']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch_size']

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    since_epoch=time.time()
    # Go through all epochs
    batch_num_all=0
    for epoch in range(0, num_epochs):

        dataset.get_epoch_similarity(epoch)
        
        # Split dataset into training and validation sets
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=False)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

        time_epoch = time.time() - since_epoch
        print('time_epoch:', time_epoch)
        utils.print_both(txt_file, 'time_epoch: {}'.format(time_epoch))
        since_epoch = time.time()
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        model.train(True)  # Set model to training mode

        running_loss = 0.0
        cur_lr = optimizer.param_groups[-1]['lr']
        print('cur_lr:', cur_lr)

        # Keep the batch number for inter-phase statistics(batch_size,15,3)
        batch_num = 1
        
        for data in train_dataloader:

            input1, input2, sim_score, _, _ = data

            input1 = input1.to(device)
            input2 = input2.to(device)
            sim_score = sim_score.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, _, _, _, _ = model(input1, input2)
                loss = criterion(outputs, sim_score*params['sim_scale'])*1e4
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * input1.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + input1.size(0))

            print_freq = 1000
            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(train_dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if writer:
                    niter = epoch * len(train_dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)

            batch_num = batch_num + 1
            batch_num_all = batch_num_all + 1

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                input1, input2, sim_score, _, _ = data

                input1 = input1.to(device)
                input2 = input2.to(device)
                sim_score = sim_score.to(device)

                outputs, _, _, _, _, _ = model(input1, input2)
                loss = criterion(outputs, sim_score*params['sim_scale'])*1e4

                val_running_loss += loss.item() * input1.size(0)

        val_loss = val_running_loss / val_size
        utils.print_both(txt_file, 'Validation Loss: {:.4f}'.format(val_loss))

        scheduler.step()

        epoch_loss = running_loss / train_size

        if epoch == 0: first_loss = epoch_loss

        if epoch == 4 and epoch_loss / first_loss > 1:
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if writer:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('Pretraining/Val_Loss' + '/Epoch', val_loss, epoch + 1)

        utils.print_both(txt_file, 'Training Loss: {:.4f}'.format(epoch_loss))

        if epoch % 5 == 0:
            pretrain_net_path = params['model_files'][-1]

            pretrain_net_epoch_folder = pretrain_net_path.replace('.pt', '')
            os.makedirs(pretrain_net_epoch_folder, exist_ok=True)
            pretrain_net_epoch_path = os.path.join(pretrain_net_epoch_folder, "model_ep%05d_l%06f.pt" % (epoch, epoch_loss))
            torch.save(model.state_dict(), pretrain_net_epoch_path)

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True

    return model

# K-means clusters initialisation
def kmeans(model, dataset, params):
    dataset.get_epoch_similarity(0)
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
    model.clustering.set_weight(weights.to(params['device']))

    return model, predicted


def Agglomerative(model,dataset, params):

    model.eval()
    dataset.get_epoch_similarity(params['epochs_pretrain'],params['epochs_pretrain'])
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