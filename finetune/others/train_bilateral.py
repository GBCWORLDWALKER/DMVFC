from __future__ import print_function, division
import numpy
import whitematteranalysis as wma
import training_functions_fiber_pair
import vtk
import mnist
import utils.fibers as fibers
import glob
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import os
import fnmatch
from utils import utils
from torch.utils.tensorboard import SummaryWriter
from test import DB_index3
from test import DiceScore
#import fibers

def list_files(input_dir,str):
    # Find input files
    input_mask = ("{0}/"+str+"*").format(input_dir)
    input_pd_fnames = glob.glob(input_mask)
    input_pd_fnames = sorted(input_pd_fnames)
    return(input_pd_fnames)
def convert_fiber_to_array(inputFile, numberOfFibers, fiberLength, numberOfFiberPoints, preproces=True,data='HCP'):
    if not os.path.exists(inputFile):
        print("<wm_cluster_from_atlas.py> Error: Input file", inputFile, "does not exist.")
        exit()
    print("\n==========================")
    print("input file:", inputFile)

    if numberOfFibers is not None:
        print("fibers to analyze per subject: ", numberOfFibers)
    else:
        print("fibers to analyze per subject: ALL")
    number_of_fibers = numberOfFibers
    fiber_length = fiberLength
    print("minimum length of fibers to analyze (in mm): ", fiber_length)
    points_per_fiber = numberOfFiberPoints
    print("Number of points in each fiber to process: ", points_per_fiber)

    # read data
    print("<wm_cluster_with_DEC.py> Reading input file:", inputFile)
    pd = wma.io.read_polydata(inputFile)

    if preproces:
        # preprocessing step: minimum length
        print("<wm_cluster_from_atlas.py> Preprocessing by length:", fiber_length, "mm.")
        pd2 = wma.filter.preprocess(pd, fiber_length, return_indices=False, preserve_point_data=True,
                                    preserve_cell_data=True, verbose=False)
    else:
        pd2 = pd

    # downsampling fibers if needed
    if number_of_fibers is not None:
        print("<wm_cluster_from_atlas.py> Downsampling to ", number_of_fibers, "fibers.")
        input_data = wma.filter.downsample(pd2, number_of_fibers, return_indices=False, preserve_point_data=True,
                                           preserve_cell_data=True, verbose=False)
    else:
        input_data = pd2

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata(input_data, points_per_fiber=args.numberOfFiberPoints)
    feat = numpy.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    feat_ROI = fiber_array.roi_list
    feat_surf_dk = fiber_array.fiber_surface_dk
    return input_data, feat, feat_ROI,feat_surf_dk
def read_data(data_dir):
    inputDir_train = data_dir
    input_pd_fnames = wma.io.list_vtk_files(inputDir_train)
    num_pd = len(input_pd_fnames)
    input_pds = []
    x_arrays=[]
    d_rois=[]
    fiber_surfs_dk = []
    for i in range(num_pd):
        input_pd,x_array,d_roi,fiber_surf_dk = \
            convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=args.numberOfFibers_train,fiberLength=args.fiberLength,
                                 numberOfFiberPoints=args.numberOfFiberPoints, preproces=False)

        # print(numpy.unique(fiber_surf_dk))
        # print(d_roi)
        fiber_surfs_dk.append(fiber_surf_dk)
        input_pds.append(input_pd)
        x_arrays.append(x_array)
        if isinstance(d_roi,list):
            d_rois.extend(d_roi)
        else:
            d_rois.append(d_roi)
    roi_unique=[]
    for d_roi in d_rois:
        roi_unique.extend(list(d_roi))
    roi_unique=numpy.unique(roi_unique)
    roi_index=numpy.array(range(len(roi_unique)))
    #roi_map=numpy.stack((roi_unique,roi_index),0)
    roi_map = numpy.load('relabel_map_nouchine.npy')

    x_arrays = numpy.array(x_arrays).reshape((-1, x_array.shape[1], x_array.shape[2]))
    ds_fs_onehot = numpy.zeros((len(x_arrays), len(numpy.unique(roi_map[1])))).astype(numpy.float32)
    if len(d_rois)==num_pd:
        ds_fs = numpy.array(d_rois).reshape((len(x_array) * num_pd, -1))
        roi_unique = numpy.unique(ds_fs)
        assert set(roi_unique).issubset(set(roi_map[0]))
        for roi in roi_unique:
            roi_new = roi_map[1][roi_map[0] == roi]
            ds_fs[ds_fs == roi] = roi_new
        for f in range(ds_fs.shape[0]):
            roi_single = numpy.unique(ds_fs[f])
            if roi_single[0] == 0:
                roi_single=roi_single[1:]
            ds_fs_onehot[f, roi_single.astype(int)] = 1
    elif len(d_rois)==x_arrays.shape[0]:
        for f,roi_fiber in enumerate(d_rois):
            roi_unique = numpy.unique(roi_fiber)
            assert set(roi_unique).issubset(set(roi_map[0]))
            for roi in roi_unique:
                roi_new = roi_map[1][roi_map[0] == roi]
                roi_fiber[roi_fiber == roi] = roi_new
            roi_single = numpy.unique(roi_fiber)
            if roi_single[0] == 0:
                roi_single=roi_single[1:]
            ds_fs_onehot[f, roi_single.astype(int)] = 1

    fiber_surfs_dk = numpy.array(fiber_surfs_dk).reshape((-1, 2))

    def surf_encoding(fiber_surf_dk,surf_map):
        fiber_surfs = fiber_surf_dk.astype(int)
        surf_labels = numpy.unique(fiber_surfs)
        for surf_label in surf_labels:
            #fiber_surfs[numpy.where(fiber_surfs == surf_label)] = numpy.where(surf_map == surf_label)
            fiber_surfs[numpy.where(fiber_surfs == surf_label)] = surf_map[1][surf_map[0] == surf_label]
        #ds_surf_onehot_dk = numpy.zeros((len(fiber_surfs), len(surf_map)))
        ds_surf_onehot_dk = numpy.zeros((len(x_arrays), len(numpy.unique(surf_map[1])))).astype(numpy.float32)
        for s in range(len(fiber_surfs)):
            ds_surf_onehot_dk[s, fiber_surfs[s]] = 1
        return ds_surf_onehot_dk
    # surf_map=numpy.unique(numpy.array(fiber_surfs_dk))
    surf_map = roi_map
    ds_surf_onehot_dk = surf_encoding(fiber_surfs_dk,surf_map)
    return  input_pds,x_arrays,ds_fs_onehot, ds_surf_onehot_dk
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
if __name__ == "__main__":
    # Translate string entries to bool for parser
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-indir',action="store", dest="inputDirectory",default="/media/annabelchen/DataShare/deepClustering/HCPTestingData/SWMA/vtk/vtk_HDYogesh_1T",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-outdir',action="store", dest="outputDirectory",default="./results",
        help='Output folder of clustering results.')
    parser.add_argument(
        '-trf', action="store", dest="numberOfFibers_train", type=int, default=None,
        help='Number of fibers of each training data to analyze from each subject.')
    parser.add_argument(
        '-l', action="store", dest="fiberLength", type=int, default=40,
        help='Minimum length (in mm) of fibers to analyze. 60mm is default.')
    parser.add_argument(
        '-p', action="store", dest="numberOfFiberPoints", type=int, default=14,
        help='Number of points in each fiber to process. 10 is default.')
    parser.add_argument('--fs', default=True, type=str2bool, help='inporparating freesurfer information')
    parser.add_argument('--surf', default=True, type=str2bool, help='inporparating cortical information')
    parser.add_argument('--ro', default=False, type=str2bool, help='outlier removal')
    parser.add_argument('--num_clusters', default=2000, type=int, help='number of clusters')
    parser.add_argument('--embedding_dimension', default=10, type=int, help='number of embeddings')
    parser.add_argument('--epochs', default=1, type=int, help='clustering epochs')
    parser.add_argument('--epochs_pretrain', default=300, type=int, help='pretraining epochs')

    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain'], help='mode')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--pretrain', default=False, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--idx', default=True, type=str2bool, help='idx for dgcnn')
    parser.add_argument('--k', default=5, type=int, help='k for dgcnn')
    parser.add_argument('--pretrained_net', default=3, help='index or path of pretrained net')
    parser.add_argument('--net_architecture', default='DGCNN', choices=['CAE_pair','DGCNN','PointNet','GCN'], help='network architecture used')
    parser.add_argument('--dataset', default='Fiber',choices=['Fiber','FiberMap'],help='custom or prepared dataset')
    parser.add_argument('--data', default='HCP',choices=['HCP', 'PPMI', 'open_fMRI'],help='custom or prepared dataset')
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
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--update_interval', default=100, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--custom_img_size', default=[128, 128, 3], nargs=3, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    args = parser.parse_args()
    print(args)

    if args.mode == 'pretrain' and not args.pretrain:
        print("Nothing to do :(")
        exit()

    board = args.tensorboard

    # Deal with pretraining option and way of showing network path
    pretrain = args.pretrain
    net_is_path = True
    if not pretrain:
        try:
            int(args.pretrained_net)
            idx = args.pretrained_net
            net_is_path = False
        except:
            pass
    params = {'pretrain': pretrain}
    # Directories
    # Create directories structure
    dirs = ['runs', 'reports', 'models']
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    import re
    # Net architecture
    model_name = args.net_architecture
    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    if pretrain or (not pretrain and net_is_path):
        reports_list = sorted(os.listdir('reports'), reverse=True)
        if reports_list:
            for file in reports_list:
                # print(file)
                if fnmatch.fnmatch(file, model_name + '*'):
                    #idx = int(str(file)[-7:-4]) + 1
                    idx=int("".join(re.findall(r'\d', file)))+1
                    break
        try:
            idx
        except NameError:
            idx = 1

    # Base filename
    name = model_name + '_' + str(idx).zfill(3)

    # Filenames for report and weights
    name_txt = name + '.txt'
    if args.pretrain:
        name_txt = name + '_pretrain.txt'
    name_net = name
    pretrained = name + '_pretrained.pt'

    # Arrange filenames for report, network weights, pretrained network weights
    if args.pretrain:
        name_txt = os.path.join('reports', name_txt)
    else:
        name_txt = os.path.join('reports', name + '_{}.txt'.format(args.epochs))
    print('report path:',  name_txt)

    name_net = os.path.join('models', name_net)
    if net_is_path and not pretrain:
        pretrained = args.pretrained_net
    else:
        pretrained = os.path.join('models', pretrained)
    if not pretrain and not os.path.isfile(pretrained):
        print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

    model_files = [name_net, pretrained]
    print(model_files)
    params['model_files'] = model_files

    # Open file
    if pretrain:
        f = open(name_txt, 'w')
    else:
        f = open(name_txt, 'a')
    params['txt_file'] = f

    # Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
    try:
        os.system("rm -rf runs/" + name)
    except:
        pass

    # Initialize tensorboard writer
    if board:
        writer = SummaryWriter('runs/' + name)
        if args.pretrain:
            writer = SummaryWriter('runs/' + name+'_pretrained')
        print('event path:', 'runs/' + name)
        params['writer'] = writer
    else:
        params['writer'] = None

    # Hyperparameters

    # Used dataset
    dataset = args.dataset

    # Batch size
    batch = args.batch_size
    params['batch'] = batch
    # Number of workers (typically 4*num_of_GPUs)
    workers = 4
    # Learning rate
    rate = args.rate
    rate_pretrain = args.rate_pretrain
    # Adam params
    # Weight decay
    weight = args.weight
    weight_pretrain = args.weight_pretrain
    # Scheduler steps for rate update
    sched_step = args.sched_step
    sched_step_pretrain = args.sched_step_pretrain
    # Scheduler gamma - multiplier for learning rate
    sched_gamma = args.sched_gamma
    sched_gamma_pretrain = args.sched_gamma_pretrain

    # Number of epochs
    epochs = args.epochs
    pretrain_epochs = args.epochs_pretrain
    params['pretrain_epochs'] = pretrain_epochs

    # Printing frequency
    print_freq = args.printing_frequency
    params['print_freq'] = print_freq

    # Clustering loss weight:
    gamma = args.gamma
    params['gamma'] = gamma

    # Update interval for target distribution:
    update_interval = args.update_interval
    params['update_interval'] = update_interval

    # Tolerance for label changes:
    tol = args.tol
    params['tol'] = tol

    # Number of clusters
    num_clusters = args.num_clusters


    # Report for settings
    tmp = "Training the '" + model_name + "' architecture"
    utils.print_both(f, tmp)
    tmp = "\n" + "The following parameters are used:"
    utils.print_both(f, tmp)
    tmp = "Batch size:\t" + str(batch)
    utils.print_both(f, tmp)
    tmp = "Number of workers:\t" + str(workers)
    utils.print_both(f, tmp)
    tmp = "Learning rate:\t" + str(rate)
    utils.print_both(f, tmp)
    tmp = "Pretraining learning rate:\t" + str(rate_pretrain)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    utils.print_both(f, tmp)
    tmp = "Pretraining weight decay:\t" + str(weight_pretrain)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler steps:\t" + str(sched_step_pretrain)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler gamma:\t" + str(sched_gamma_pretrain)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
    utils.print_both(f, tmp)
    tmp = "Clustering loss weight:\t" + str(gamma)
    utils.print_both(f, tmp)
    tmp = "Update interval for target distribution:\t" + str(update_interval)
    utils.print_both(f, tmp)
    tmp = "Stop criterium tolerance:\t" + str(tol)
    utils.print_both(f, tmp)
    tmp = "Number of clusters:\t" + str(num_clusters)
    utils.print_both(f, tmp)
    tmp = "Leaky relu:\t" + str(args.leaky)
    utils.print_both(f, tmp)
    tmp = "Leaky slope:\t" + str(args.neg_slope)
    utils.print_both(f, tmp)
    tmp = "Activations:\t" + str(args.activations)
    utils.print_both(f, tmp)
    tmp = "Bias:\t" + str(args.bias)
    utils.print_both(f, tmp)

    # Data preparation
    if dataset == 'Fiber':
        data_dir = args.inputDirectory
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)
        input_pds,x_arrays,x_roi,x_surf_dk=read_data(data_dir)
        num_points = args.numberOfFiberPoints
        tmp = "numner of points used: {}".format(num_points)
        utils.print_both(f, tmp)
        dataset = mnist.FiberPair(x_arrays, x_roi, x_surf_dk, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch, shuffle=True, num_workers=workers)
        dataloader_entire = torch.utils.data.DataLoader(dataset, batch_size=int(batch/2), shuffle=False, num_workers=workers)
        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
    elif dataset == 'FiberMap':
        data_dir = args.inputDirectory
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)
        input_pds,x_arrays,x_roi,x_surf_dk=read_data(data_dir)
        img_size = [28, 28, 3]
        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)
        import mnist
        dataset = mnist.FiberMap_pair(x_arrays,x_roi,x_surf_dk,transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch, shuffle=False, num_workers=workers)
        dataloader_entire=dataloader
        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

    params['dataset_size'] = dataset_size

    # GPU check
    #device="cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    tmp = "\nPerforming calculations on:\t" + str(device)
    utils.print_both(f, tmp + '\n')
    params['device'] = device

    if args.net_architecture == 'DGCNN':
        if args.idx:
            idxf=torch.zeros((num_points,args.k),dtype=torch.int64,device=device)
            if args.k==5:
                idxf[:,0]=torch.tensor(range(num_points))
                idxf[:, 1]=torch.tensor(range(num_points))-2
                idxf[:, 2] = torch.tensor(range(num_points)) - 1
                idxf[:, 3] = torch.tensor(range(num_points)) + 1
                idxf[:, 4] = torch.tensor(range(num_points)) + 2
            elif args.k==3:
                idxf[:,0]=torch.tensor(range(num_points))
                idxf[:, 1] = torch.tensor(range(num_points)) - 1
                idxf[:, 2] = torch.tensor(range(num_points)) + 1
            idxf[idxf<0]=0
            idxf[idxf>num_points-1]=num_points-1
            idx=idxf.repeat(batch,1,1)
        else:
            idx=None


    # Evaluate the proper model
    if args.net_architecture=='CAE_pair':
        to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters,embedding_dimension=args.embedding_dimension, leaky = args.leaky, neg_slope = args.neg_slope)"
    elif args.net_architecture=='DGCNN' :
        to_eval = "nets." + model_name + "(k=args.k,input_channel=3,num_clusters=num_clusters,embedding_dimension=args.embedding_dimension,idx=idx)"
    elif args.net_architecture == 'PointNet' or args.net_architecture == 'GCN':
        to_eval = "nets." + model_name + "(input_channel=3,num_clusters=num_clusters,embedding_dimension=args.embedding_dimension)"

    model = eval(to_eval)

    # Tensorboard model representation
    # if board:
    #     writer.add_graph(model, torch.autograd.Variable(torch.Tensor(batch, img_size[2], img_size[0], img_size[1])))

    model = model.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_pa=count_parameters(model)
    print('number of parameters:',num_pa)

    # Reconstruction loss
    criterion_1 = nn.MSELoss(size_average=True)
    # Clustering loss
    criterion_2 = nn.KLDivLoss(size_average=False)

    criteria = [criterion_1, criterion_2]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)

    optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain, weight_decay=weight_pretrain)

    optimizers = [optimizer, optimizer_pretrain]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain, gamma=sched_gamma_pretrain)

    schedulers = [scheduler, scheduler_pretrain]

    if args.mode == 'train_full':
        model_pretrained, model,preds_initial, preds_final,probs_final= training_functions_fiber_pair.train_model(model, dataloader, dataloader_entire, criteria, optimizers, schedulers, epochs, params,x_roi,args.fs,x_surf_dk,args.surf)

        if args.fs:
            if args.surf:
                name_net_save = name_net + '_{}_fs_surf.pt'.format(epochs)
            else:
                name_net_save=name_net + '_{}_fs.pt'.format(epochs)
        else:
            name_net_save = name_net + '_{}.pt'.format(epochs)
        torch.save(model.state_dict(), name_net_save)
        print(name_net_save)
        def roi_cluster_uptate(num_clusters, preds, x_fs):
            roi_cluster = numpy.zeros([num_clusters, x_fs.shape[1]])
            for i in range(num_clusters):
                t = x_fs[preds == i]
                t1 = numpy.sum(t, 0)
                roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
                if 0 in roi_all:
                    roi_all = roi_all[1:]
                roi_cluster[i, roi_all] = 1
            return roi_cluster
        def surf_cluster_uptate(num_clusters, preds, x_surf):
            surf_cluster = numpy.zeros([num_clusters, x_surf.shape[1]])
            for i in range(num_clusters):
                t = x_surf[preds == i]
                t1 = numpy.sum(t, 0)
                surf_cluster[i] = t1 / t1.sum()
            return surf_cluster

        if args.fs:
            roi_cluster = roi_cluster_uptate(model.num_clusters, preds_final, x_roi)
            #numpy.save('utils/roi_cluster.npy', roi_cluster)
        if args.surf:
            surf_cluster = surf_cluster_uptate(model.num_clusters, preds_final, x_surf_dk)
            #numpy.save('utils/surf_cluster.npy', surf_cluster)


        def metrics_calculation(predicted, x_arrays, x_fs, x_surf):
            loss_fn = DiceScore()

            def tapc_calculation1(num_clusters, preds, roi_fs):
                roi_cluster = numpy.zeros([num_clusters, roi_fs.shape[1]])
                tapc_all = []
                for i in range(num_clusters + 1):
                    t = roi_fs[preds == i]
                    if t.size == 0:
                        continue
                    else:
                        t1 = numpy.sum(t, 0)
                        roi_all = numpy.where(t1 > t.shape[0] * 0.4)[0]
                        if 0 in roi_all:
                            roi_all = roi_all[1:]
                        roi_cluster[i, roi_all] = 1
                        roi_preds = numpy.repeat(roi_cluster[i].reshape((1, len(roi_cluster[i]))), t.shape[0], axis=0)
                        tapc = loss_fn(t, roi_preds)
                        tapc_all.append(tapc)
                # roi_preds = roi_cluster[preds]
                # tapc = loss_fn(roi_fs, roi_preds)
                tapc = numpy.mean(tapc_all)
                return tapc, numpy.array(tapc_all)

            def tspc_calculation1(num_clusters, preds, ds_surf_onehot):
                tspc_sub = []
                N_surf_all = []
                for i in range(num_clusters + 1):
                    t = ds_surf_onehot[preds == i]
                    if t.size == 0:
                        continue
                    else:
                        t1 = numpy.sum(t, 0)
                        if t1.sum() == 0:
                            continue
                        surf_cluster = t1 / t1.sum()
                        tspc_all = surf_cluster * t
                        tspc1 = numpy.sum(tspc_all, 1)
                        tspc_clu = numpy.mean(tspc1)

                        surf_all = numpy.where(t1 > 0)[0]
                        N_surf_all.append(len(surf_all))
                        tspc_sub.append(tspc_clu)
                tspc = numpy.array(tspc_sub).mean()
                N_surf_all = numpy.array(N_surf_all).mean()
                return tspc, N_surf_all, numpy.array(tspc_sub)

            tapc_train, tapc_all = tapc_calculation1(num_clusters, predicted, x_fs)
            tspc_train, _, tspc_all = tspc_calculation1(num_clusters, predicted, x_surf)
            # print(x_arrays.shape)
            # print(predicted.shape)
            DB_score, DB_all, dis_intra, dis_inter = DB_index3(x_arrays, predicted)
            n_detected = 0
            flag_detected = numpy.zeros(num_clusters)
            for n in range(num_clusters):
                n_fiber = numpy.sum(predicted == n)
                if n_fiber >= 20:
                    n_detected += 1
                    flag_detected[n] = 1
            wmpg_train = n_detected / num_clusters
            utils.print_both(f, 'DB: {0:.4f}\tWMPG: {1:.4f}\tTAPC: {2:.4f}\tTSPC: {3:.4f}'.format(DB_score, wmpg_train,
                                                                                                  tapc_train,
                                                                                                  tspc_train))
            return DB_score, wmpg_train, tapc_train, tspc_train
        metrics_calculation(preds_final, x_arrays, x_roi, x_surf_dk)
        if args.ro:
            num_std = 1.5
            preds_fs_surf = preds_final
            probs_fs_surf = probs_final

            id_reject = []
            probs_rejected = []
            for ic in range(num_clusters):
                index = numpy.where(preds_fs_surf == ic)[0]
                probc = probs_fs_surf[index]
                if len(probc) > 0:
                    index1 = numpy.where((probc.mean() - probc) > num_std * probc.std())[0]
                    if len(index1) > 0:
                        id_rejectc = index[index1]
                        id_reject.extend(id_rejectc)
                        prob_rejected = probc[index1]
                        probs_rejected.append(prob_rejected)
            id_reject = numpy.array(id_reject)
            if id_reject is not None:
                temp = numpy.ones(len(preds_fs_surf))
                temp[id_reject] = 0
                mask = temp > 0
                x_array_ro = x_arrays[mask]
                preds_fs_surf_ro = preds_fs_surf[mask]
                probs_fs_surf_ro = probs_fs_surf[mask]
                ds_fs_onehot_ro = x_roi[mask]
                ds_surf_onehot_ro = x_surf_dk[mask]
                rate_left = 1 - len(preds_fs_surf_ro) / len(preds_fs_surf)
                print('fiber_removed:', rate_left)

                metrics_calculation(preds_fs_surf_ro , x_array_ro, ds_fs_onehot_ro, ds_surf_onehot_ro)
                outdir = args.outputDirectory
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                print('<train.py> Saving output cluster files in directory:', outdir)
                maskc = numpy.ones(len(preds_final))
                maskc[id_reject] = 0
                print(input_pds[0].GetNumberOfLines(), maskc.shape)
                pd_c = wma.filter.mask(input_pds[0], maskc, preserve_point_data=True, preserve_cell_data=True, verbose=False)

                output_kept = os.path.join(outdir, 'all', 'all.vtp')
                if not os.path.exists(os.path.join(outdir, 'all')):
                    os.makedirs(os.path.join(outdir, 'all'))
                wma.io.write_polydata(pd_c, output_kept)

                maskc = numpy.zeros(len(preds_final))
                maskc[id_reject] = 1
                pd_rej = wma.filter.mask(input_pds[0], maskc, preserve_point_data=True, preserve_cell_data=True, verbose=False)
                output_rej = os.path.join(outdir, 'rejected', 'rejected.vtp')
                if not os.path.exists(os.path.join(outdir, 'rejected')):
                    os.makedirs(os.path.join(outdir, 'rejected'))
                wma.io.write_polydata(pd_rej, output_rej)

                print('<train.py> %d / %f fibers are removed:' % (id_reject.shape[0], len(preds_final)))
                
                def add_prob(inpd, prob):
                    vtk_array = vtk.vtkDoubleArray()
                    vtk_array.SetName('Prob')
                    inpd.GetLines().InitTraversal()
                    for lidx in range(0, inpd.GetNumberOfLines()):
                        ptids = vtk.vtkIdList()
                        inpd.GetLines().GetNextCell(ptids)
                        prob_line = prob[lidx]
                        for pidx in range(0, ptids.GetNumberOfIds()):
                            vtk_array.InsertNextTuple1(prob_line)
                    inpd.GetPointData().AddArray(vtk_array)
                    return inpd
                input_pd1 = add_prob(pd_c, probs_fs_surf_ro)
                cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
                pd_c_list = wma.cluster.mask_all_clusters(input_pd1, preds_fs_surf_ro,
                                                          num_clusters,
                                                          preserve_point_data=True,
                                                          preserve_cell_data=True,
                                                          verbose=False)

                cluster_sizes = list()
                cluster_fnames = list()
                fnames = list()
                # cluster_colors = list()
                for c in range(num_clusters):
                    mask = preds_fs_surf_ro == c
                    cluster_size = numpy.sum(mask)
                    cluster_sizes.append(cluster_size)
                    # pd_c = wma.filter.mask(output_polydata_s, mask, preserve_point_data=True, preserve_cell_data=True,verbose=False)
                    pd_c = pd_c_list[c]
                    # The clusters are stored starting with 1, not 0, for user friendliness.
                    fname_c = 'cluster_{0:05d}.vtp'.format(c + 1)
                    # save the filename for writing into the MRML file
                    fnames.append(fname_c)
                    # prepend the output directory
                    fname_c = os.path.join(outdir, fname_c)
                    cluster_fnames.append(fname_c)
                    wma.io.write_polydata(pd_c, fname_c)

                # Notify user if some clusters empty
                print(
                    "<wm_cluster_atlas.py> Checking for empty clusters (can be due to anatomical variability or too few fibers analyzed).")
                for sz, fname in zip(cluster_sizes, cluster_fnames):
                    if sz == 0:
                        print(sz, ":", fname)

                cluster_sizes = numpy.array(cluster_sizes)
                print("<wm_cluster_from_atlas.py> Mean number of fibers per cluster:", numpy.mean(cluster_sizes),
                      "Range:",
                      numpy.min(cluster_sizes), "..", numpy.max(cluster_sizes))
                # Also write one with 100%% of fibers displayed
                fname = os.path.join(outdir, 'clustered_tracts_display_100_percent.mrml')
                wma.mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)
                render = True
                # View the whole thing in png format for quality control
                # if render:

                #     try:
                #         print('<wm_cluster_from_atlas.py> Rendering and saving images of clustered subject.')
                #         ren = wma.render.render(input_pd1, 1000, data_mode='Cell', data_name='EmbeddingColor',
                #                                 verbose=False)
                #         ren.save_views(outdir)
                #         del ren
                #     except:
                #         print('<wm_cluster_from_atlas.py> No X server available.')

                print("\n==========================")
                print('<wm_cluster_from_atlas.py> Done clustering subject.  See output in directory:\n ', outdir, '\n')


                # save outlier fibers
                outdir = args.outputDirectory + '/outliers'
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                print('<wm_cluster_atlas.py> Saving outlier cluster files in directory:', outdir)
                maskc = numpy.zeros(len(preds_final))
                maskc[id_reject] = 1
                pd_c = wma.filter.mask(input_pds[0], maskc, preserve_point_data=True, preserve_cell_data=True,
                                       verbose=False)

                input_pd1 = add_prob(pd_c, probs_fs_surf_rej)
                cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
                pd_c_list = wma.cluster.mask_all_clusters(input_pd1, preds_fs_surf_rej,
                                                          num_clusters,
                                                          preserve_point_data=True,
                                                          preserve_cell_data=True,
                                                          verbose=False)

                cluster_sizes = list()
                cluster_fnames = list()
                fnames = list()
                # cluster_colors = list()
                for c in range(num_clusters):
                    mask = preds_fs_surf_rej == c
                    cluster_size = numpy.sum(mask)
                    cluster_sizes.append(cluster_size)
                    # pd_c = wma.filter.mask(output_polydata_s, mask, preserve_point_data=True, preserve_cell_data=True,verbose=False)
                    pd_c = pd_c_list[c]
                    # The clusters are stored starting with 1, not 0, for user friendliness.
                    fname_c = 'cluster_{0:05d}.vtp'.format(c + 1)
                    # save the filename for writing into the MRML file
                    fnames.append(fname_c)
                    # prepend the output directory
                    fname_c = os.path.join(outdir, fname_c)
                    cluster_fnames.append(fname_c)
                    wma.io.write_polydata(pd_c, fname_c)

                # Notify user if some clusters empty
                print(
                    "<wm_cluster_atlas.py> Checking for empty clusters (can be due to anatomical variability or too few fibers analyzed).")
                for sz, fname in zip(cluster_sizes, cluster_fnames):
                    if sz == 0:
                        print(sz, ":", fname)

                cluster_sizes = numpy.array(cluster_sizes)
                print("<wm_cluster_from_atlas.py> Mean number of fibers per cluster:", numpy.mean(cluster_sizes),
                      "Range:",
                      numpy.min(cluster_sizes), "..", numpy.max(cluster_sizes))
                # Also write one with 100%% of fibers displayed
                fname = os.path.join(outdir, 'clustered_tracts_display_100_percent.mrml')
                wma.mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=0.1)
                render = True
                # # View the whole thing in png format for quality control
                # if render:

                #     try:
                #         print('<wm_cluster_from_atlas.py> Rendering and saving images of clustered subject.')
                #         ren = wma.render.render(input_pd1, 1000, data_mode='Cell', data_name='EmbeddingColor',
                #                                 verbose=False)
                #         ren.save_views(outdir)
                #         del ren
                #     except:
                #         print('<wm_cluster_from_atlas.py> No X server available.')

                print("\n==========================")
                print('<wm_cluster_from_atlas.py> Done clustering subject.  See output in directory:\n ', outdir, '\n')

        else:
            def fiber_cluster_save(input_pds,predicted):
                cluster_colors = numpy.random.randint(0, 255, (num_clusters, 3))
                appender = vtk.vtkAppendPolyData()
                for pd in input_pds:
                    if (vtk.vtkVersion().GetVTKMajorVersion() >= 6.0):
                        appender.AddInputData(pd)
                    else:
                        appender.AddInput(pd)
                appender.Update()
                input_data = appender.GetOutput()
                outdir = args.outputDirectory
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                print('outdir:', outdir)
                pd_c_list = wma.cluster.mask_all_clusters(input_data, predicted,
                                                                                            num_clusters,
                                                                                            preserve_point_data=True,
                                                                                            preserve_cell_data=True,
                                                                                            verbose=False)
                print('<wm_cluster_atlas.py> Saving output cluster files in directory:', outdir)
                cluster_sizes = list()
                cluster_fnames = list()
                fnames = list()
                for c in range(num_clusters):
                    mask = predicted == c
                    cluster_size = numpy.sum(mask)
                    cluster_sizes.append(cluster_size)
                    pd_c = pd_c_list[c]
                    # The clusters are stored starting with 1, not 0, for user friendliness.
                    fname_c = 'cluster_{0:05d}.vtp'.format(c + 1)
                    # save the filename for writing into the MRML file
                    fnames.append(fname_c)
                    # prepend the output directory
                    fname_c = os.path.join(outdir, fname_c)
                    cluster_fnames.append(fname_c)
                    wma.io.write_polydata(pd_c, fname_c)

                # Notify user if some clusters empty
                print(
                    "<wm_cluster_atlas.py> Checking for empty clusters (can be due to anatomical variability or too few fibers analyzed).")
                for sz, fname in zip(cluster_sizes, cluster_fnames):
                    if sz == 0:
                        print(sz, ":", fname)

                cluster_sizes = numpy.array(cluster_sizes)
                print("<wm_cluster_from_atlas.py> Mean number of fibers per cluster:", numpy.mean(cluster_sizes),
                      "Range:",
                      numpy.min(cluster_sizes), "..", numpy.max(cluster_sizes))
                # Also write one with 100%% of fibers displayed
                fname = os.path.join(outdir, 'clustered_tracts_display_100_percent.mrml')
                wma.mrml.write(fnames, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)
                render = True
                # View the whole thing in png format for quality control
                if render:

                    try:
                        print('<wm_cluster_from_atlas.py> Rendering and saving images of clustered subject.')
                        ren = wma.render.render(input_data, 1000, data_mode='Cell', data_name='EmbeddingColor',
                                                verbose=False)
                        ren.save_views(outdir)
                        del ren
                    except:
                        print('<wm_cluster_from_atlas.py> No X server available.')

                print("\n==========================")
                print('<wm_cluster_from_atlas.py> Done clustering subject.  See output in directory:\n ', outdir, '\n')
            fiber_cluster_save(input_pds, preds_final)

    elif args.mode == 'pretrain':
        model = training_functions_fiber_pair.pretraining(model, dataloader, criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
        print(name)

    # Close files
    f.close()
    if board:
        writer.close()
