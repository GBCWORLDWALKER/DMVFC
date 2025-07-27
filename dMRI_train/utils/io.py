import os
import numpy
import fnmatch
import re
import vtk
import h5py
import glob
import whitematteranalysis as wma
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import colorsys

from utils.fibers import FiberArray
from utils.utils import add_vtk_arry, surf_encoding, print_both, convert_to_polydata, fiber_reorder
import wandb

def process_args_test(args):

    params = {}


    params['outputDirectory'] = args.outputDirectory
    if params['outputDirectory'].endswith('/'):
        params['outputDirectory'] = params['outputDirectory'][:-1]
    params['outputDirectory'] = params['outputDirectory']

    os.makedirs(params['outputDirectory'], exist_ok=True)
    params['trained_net'] = args.trained_net
    # params['trained_net'] = max(glob.glob(args.trained_net + "DGCNN_*_final_k10.pt"), key=os.path.getctime)
    params['atlas_pred'] = args.atlas_pred
    params['subID'] ='1'
    # pretain or load previous trained pretrained model
    params['funct'] = args.funct
    params["output_clusters"] = args.output_clusters
    params['batch_size'] = args.batch_size
    params["num_points"] = args.numberOfFiberPoints
    params["ro_std"] = args.ro_std
    params["surf"] = args.surf
    report_file = os.path.join(params['outputDirectory'], "log.txt")
    params['report_file'] = report_file
    params['log'] = open(params['report_file'], 'w')
    params['inputDirectory'] = args.inputDirectory
 
    params['gpu'] = args.gpu
    params['wandb_group'] = args.wandb_group
    params['bundle'] = args.bundle
    return params

def process_args(args):
    params = {}

    # input and output dir
    params['inputDirectory'] = args.inputDirectory
    params['outputDirectory'] = args.outputDirectory
    params['fmri_path'] = args.fmri_path
    if params['outputDirectory'].endswith('/'):
        params['outputDirectory'] = params['outputDirectory'][:-1]
    if args.bundle_list_path is not None:
        params['bundle_list_path'] = args.bundle_list_path
    if args.index_path is not None:
        params['index_path'] = args.index_path

    # pretain or load previous trained pretrained model
    params["pretrain"] = args.pretrain
    params["output_clusters"] = args.output_clusters
    params["net_architecture"] = args.net_architecture
    params["num_clusters"] = args.num_clusters
    params['batch_size'] = args.batch_size
    params["num_points"] = args.numberOfFiberPoints
    params["embedding_dimension"] = args.embedding_dimension
    params["reclustering"] = args.reclustering
    params["ro"] = args.ro
    params["ro_std"] = args.ro_std

    # Clustering loss weight
    params['gamma'] = args.gamma

    # embedding and clustering loss
    params['embedding_surf'] = args.embedding_surf
    params['loss_surf'] = args.loss_surf
    params['loss_subID'] = args.loss_subID
    params['full_brain'] = args.full_brain
    params['clustering_fiber_interval'] = args.clustering_fiber_interval
    params['GPU'] = args.GPU
    # Freeze embedding weights or not
    params['freeze'] = args.freeze
    params['alpha'] = args.alpha
    # Update interval for target distribution:
    params['update_interval'] = args.update_interval
    params['funct'] = args.funct
    # Tolerance for label changes:
    params['tol'] = args.tol
    params['dmri_similarity_path'] = args.dmri_similarity_path
    # Number of epochs
    params['epochs'] = args.epochs
    params['epochs_pretrain'] = args.epochs_pretrain

    # Learning rate
    params['rate'] = args.rate
    params['rate_pretrain'] = args.rate_pretrain
    params['bundle'] = args.bundle
    # Adam params
    # Weight decay
    params['weight'] = args.weight
    params['weight_pretrain'] = args.weight_pretrain

    # Scheduler steps for rate update
    params['sched_step'] = args.sched_step
    params['sched_step_pretrain'] = args.sched_step_pretrain

    # Scheduler gamma - multiplier for learning rate
    params['sched_gamma'] = args.sched_gamma
    params['sched_gamma_pretrain'] = args.sched_gamma_pretrain

    # Printing frequency
    params['print_freq'] = args.printing_frequency
    params["fmri_similarity_path"] = args.fmri_similarity_path
    params["dmri_similarity_path"] = args.dmri_similarity_path
    params["similarity_path"] = args.similarity_path

    # New params
    params['dataset_prepared'] = args.dataset_prepared
    
    wandb_group = args.wandb_group
    if wandb_group is not None:
        os.environ["WANDB_RUN_GROUP"] = wandb_group
        wandb.init(group=wandb_group)
    else:
        pass

    # Create directories structure
    dirs = [os.path.join(args.outputDirectory, 'runs'), os.path.join(args.outputDirectory, 'reports'),
            os.path.join(args.outputDirectory, 'models')]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # name of the trained model
    model_name = params["net_architecture"]
    params['model_name'] = model_name

    # output indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    reports_list = sorted(os.listdir(os.path.join(args.outputDirectory, 'reports')), reverse=True)
    if reports_list:
        for file in reports_list:
            if fnmatch.fnmatch(file, model_name + '*'):
                saveidx = int("".join(re.findall(r'\d', file))) + 1
                break
    else:
        saveidx = 1

    # name for this run, which will be the base filename in the output
    if args.output_name is not None:
        run_name = args.output_name
    else:
        run_name = model_name + '_' + str(saveidx).zfill(3)
    params['run_name'] = run_name


    output_name = "DFC_" + os.path.basename(params["outputDirectory"]) + "_" + params["run_name"] + '_' + \
                  'EmbEP%s' % params['embedding_surf'] + '_' + 'CluEP%s' % params['loss_surf'] + '_' \
                  'gamma%s' % params["gamma"] + '_ro%s' % params["ro_std"] + '_k%s' % params["num_clusters"]
    params['output_name'] = output_name

    output_dir = os.path.join(params["outputDirectory"], output_name)
    params['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # output files of the trained models
    if args.pretrain:
        trained_pretrain_net_path = "None"
    else:
        trained_pretrain_net_path = args.pretrained_net
        if not os.path.isfile(trained_pretrain_net_path):
            print("Error: No pretrained weights at %s" % trained_pretrain_net_path)
            exit()
    
    wandb_group = args.wandb_group
    if wandb_group is not None:
        os.environ["WANDB_RUN_GROUP"] = wandb_group
        wandb.init(group=wandb_group)
        wandb.config.update({
        "pretrain_net_path": os.path.join(args.outputDirectory, 'models', f'{run_name}_pretrain_k{args.num_clusters}.pt'),
        "final_net_path": os.path.join(args.outputDirectory, 'models', f'{run_name}_final_k{args.num_clusters}.pt')
    })
        print("wandb.config['alpha']: ", wandb.config['alpha'])
    else:
        pretrain_net_path = os.path.join(args.outputDirectory, 'models', f'{run_name}_pretrain_k{args.num_clusters}.pt')
        final_net_path = os.path.join(args.outputDirectory, 'models', f'{run_name}_final_k{args.num_clusters}.pt')
    # Store model paths as config variables in wandb
    
    if wandb_group is not None:
        # Assign the paths to variables for use in the current run
        pretrain_net_path = wandb.config.pretrain_net_path
        final_net_path = wandb.config.final_net_path
    model_files = [trained_pretrain_net_path, pretrain_net_path, final_net_path]
    params['model_files'] = model_files

    # Setup log
    report_file = os.path.join(args.outputDirectory, 'reports', run_name + ".txt")
    params['report_file'] = report_file
    params['log'] = open(params['report_file'], 'w')

    # Initialize tensorboard writer
    if args.tensorboard:
        # Delete tensorboard entry if exist (not to overlap as the charts become unreadable)
        try:
            os.system("rm -rf %s/runs/%s" % (args.outputDirectory, run_name))
        except:
            pass

        writer = SummaryWriter('%s/runs/%s' % (args.outputDirectory, run_name))
        params['writer'] = writer
        params['tensorboard_path'] = '%s/runs/%s' % (args.outputDirectory, run_name)
    else:
        params['writer'] = None

    return params

def numsort(x):
    num=re.findall(r'\d+',x)
    return int(num[0])
def read_data(params, verbose=False):
    # Iterate through subject subfolders
    full_array=[]
    full_pd=[]
    index=0
    for subject_folder in sorted(os.listdir(params["inputDirectory"]), key=numsort):
        subject_path = os.path.join(params["inputDirectory"], subject_folder)
        if os.path.isdir(subject_path):
            try:
                vtk_files = wma.io.list_vtk_files(subject_path)
                if len(vtk_files) == 0:
                    print(f"Warning: No VTK files found in {subject_folder}")
                    continue
                
                # Process all VTK files in the subject folder
                subject_input_pd_list = []
                subject_fiber_array_list = []
                for vtk_file in vtk_files:
                    if params['bundle'] is not None:
                        filename_to_check = os.path.basename(vtk_file)
                        target_bundle = params['bundle']
                        
                        # 使用正则表达式确保 target_bundle 是文件名中的一个完整部分，
                        # 而不是更长名称的子串。
                        # 它应该被非字母数字字符或文件名的开头/结尾所包围。
                        pattern = r"(^|[^a-zA-Z0-9])" + re.escape(target_bundle) + r"($|[^a-zA-Z0-9])"
                        
                        if not re.search(pattern, filename_to_check):
                            continue
                    input_pd, fiber_array = convert_pd_to_array(vtk_file, numberOfFiberPoints=params["num_points"], verbose=verbose)
                    subject_input_pd_list.append(input_pd)
                    subject_fiber_array_list.append(fiber_array)
                    fiber_array.fiber_subID = numpy.full(len(fiber_array.fiber_array_ras), int(index))
                    index+=1
                # Combine the processed data for this subject
                if subject_input_pd_list:
                    subject_input_pd = vtk.vtkAppendPolyData()
                    for pd in subject_input_pd_list:
                        subject_input_pd.AddInputData(pd)
                    subject_input_pd.Update()
                    subject_input_pd = subject_input_pd.GetOutput()
                    subject_fiber_array = FiberArray()
                    subject_fiber_array.combine_fiber_arrays(subject_fiber_array_list)
                    full_array.append(subject_fiber_array)
                    full_pd.append(subject_input_pd)



            except Exception as e:
                print(f"Error processing {subject_folder}: {str(e)}")
    full_array_fiber = FiberArray()
    full_array_fiber.combine_fiber_arrays(full_array)
    full_pd_fiber = vtk.vtkAppendPolyData()
    for pd in full_pd:
        full_pd_fiber.AddInputData(pd)
    full_pd_fiber.Update()
    full_pd_fiber = full_pd_fiber.GetOutput()
    # roi_map = numpy.load('./resources/relabel_map_hHOA.npy')
    # roi_map = numpy.load('./resources/relabel_map_mHOA2.npy')
    # combined_fiber_array.fiber_array_endpoints_onehot = surf_encoding(combined_fiber_array.fiber_endpoint, roi_map)

    return full_pd_fiber, full_array_fiber


def convert_pd_to_array(inputFile, numberOfFiberPoints, numberOfFibers=None, fiberLength=None, preproces=False, verbose=False):
    
    if not os.path.exists(inputFile):
        print("Error: Input file", inputFile, "does not exist.")
        exit()

    if numberOfFibers is not None:
        print("fibers to analyze per subject: ", numberOfFibers)
    else:
        print("fibers to analyze per subject: ALL")

    fiber_length = fiberLength
    print("minimum length of fibers to analyze (in mm): ", fiber_length)
    points_per_fiber = numberOfFiberPoints
    print("number of points in each fiber to process: ", points_per_fiber)

    # read data
    print("Reading input file:", inputFile)
    pd = wma.io.read_polydata(inputFile)

    if preproces:
        # preprocessing step: minimum length
        print("Preprocessing by length:", fiber_length, "mm.")
        pd2 = wma.filter.preprocess(pd, fiber_length, 
                                    return_indices=False, preserve_point_data=True,
                                    preserve_cell_data=True, verbose=False)
    else:
        pd2 = pd

    # downsampling fibers if needed
    if numberOfFibers is not None:
        print("<wm_cluster_from_atlas.py> Downsampling to ", numberOfFibers, "fibers.")
        input_pd = wma.filter.downsample(pd2, numberOfFibers,
                                           return_indices=False, preserve_point_data=True,
                                           preserve_cell_data=True, verbose=False)
    else:
        input_pd = pd2

    fiber_array = FiberArray()
    fiber_array.convert_from_polydata(input_pd, points_per_fiber=numberOfFiberPoints, verbose=verbose)
    
    return input_pd, fiber_array

import random
def generate_distinct_colors(n):
    cmap = plt.get_cmap('hsv')
    colors = [cmap(i / n) for i in range(n)]
    # Shuffle the colors to avoid adjacent similar colors
    np.random.shuffle(colors)
    # Convert to RGB tuples
    return [(r, g, b) for r, g, b, _ in colors]

def output_data(fiber_subID, params, input_pd, preds_final, probs_final, surf_cluster, df_stats, cluster_centroids, cluster_reordered_fibers, outlier_fibers=None, clusters_to_remove=[], freorder=False, verbose=False):
    output_name = params['output_name']
    output_dir = params['output_dir']

    preds_h5 = os.path.join(output_dir, output_name+"_pred.h5")
    with h5py.File(preds_h5, "w") as f:
        f.create_dataset('preds_final', data=preds_final)
        f.create_dataset('probs_final', data=probs_final)
        if surf_cluster is not None:
            f.create_dataset('surf_cluster', data=surf_cluster.detach().cpu().numpy())

    # output cluster centroids
    centroid_vtk = os.path.join(output_dir, output_name+"_centroid.vtp")
    if outlier_fibers is not None:
        centroid_vtk = os.path.join(output_dir, output_name + "_ro_centroid.vtp")
    add_arrays = {}
    add_arrays["ClusterUratio"] = [float(v[:5]) for v in df_stats["Uratio"].values.tolist()]
    add_arrays["ClusterNoF"] = [float(v) for v in df_stats["NoFiber"].values.tolist()]
    add_arrays["ClusterProb"] = [float(v[:5]) for v in df_stats["Pred-Prob"].values.tolist()]
    add_arrays["ClusterDB"] = [float(v) for v in df_stats["DB-score"].values.tolist()]
    add_arrays["ClusterAlpha"] = [float(v) for v in df_stats["Alpha"].values.tolist()]
    add_arrays["ClusterTSPC"] = [float(v) for v in df_stats["TSPC"].values.tolist()]
    add_arrays["ClusterWMPG"] = [float(v) for v in df_stats["WMPG"].values.tolist()]
    # add_arrays["SubClusterNoF"] = [float(v[:5]) for v in df_stats["Sub-NoFiber"].values.tolist()]
    # add_arrays["SubClusterUratio"] = [float(v[:5]) for v in df_stats["Sub-Uratio"].values.tolist()]
    # add_arrays["SubClusterProb"] = [float(v[:5]) for v in df_stats["Sub-Pred-Prob"].values.tolist()]
    # add_arrays["SubClusterDB"] = [float(v[:5]) for v in df_stats["Sub-DB-score"].values.tolist()]
    # add_arrays["SubClusterAlpha"] = [float(v[:5]) for v in df_stats["Sub-Alpha"].values.tolist()]
    # add_arrays["SubClusterTSPC"] = [float(v[:5]) for v in df_stats["Sub-TSPC"].values.tolist()]
    pd_centroid = convert_to_polydata(cluster_centroids, arrays=add_arrays)
    wma.io.write_polydata(pd_centroid, centroid_vtk)
    print_both(params["log"], "Save centroid vtk file at:\t %s." % centroid_vtk)

    if outlier_fibers is None: # no use to output initial clusters
        print_both(params["log"], "Not vtk file output because no outlier indices provided")
        return

    if not params["output_clusters"]:
        return

    # output fiber cluster files
    num_clusters = params.get('num_clusters', len(numpy.unique(preds_final)) - 1)  # Subtract 1 to account for outliers
    cluster_folder = os.path.join(output_dir)
    outlier_folder = os.path.join(cluster_folder, 'outliers')
    rej_cluster_folder = os.path.join(cluster_folder, 'rejected_clusters')
    os.makedirs(outlier_folder, exist_ok=True)
    os.makedirs(rej_cluster_folder, exist_ok=True)

    preds_final[outlier_fibers] = preds_final[outlier_fibers] + params["num_clusters"]
    pd_c_list = wma.cluster.mask_all_clusters(input_pd, preds_final, num_clusters*2, preserve_point_data=True, preserve_cell_data=False, verbose=False)

    appender = vtk.vtkAppendPolyData()
    for c_idx in range(num_clusters * 2):
        pd_c = pd_c_list[c_idx]

        if c_idx < num_clusters:

            pd_c = add_vtk_arry(pd_c, "FiberProb", probs_final[preds_final == c_idx])
            if freorder:
                reorder = cluster_reordered_fibers[c_idx]
                pd_c = fiber_reorder(pd_c, reorder)

            if c_idx not in clusters_to_remove:
                fname_c = os.path.join(cluster_folder, 'cluster_{0:05d}.vtp'.format(c_idx + 1))
                wma.io.write_polydata(pd_c, fname_c)
            else:
                fname_c = os.path.join(rej_cluster_folder, 'cluster_{0:05d}.vtp'.format(c_idx + 1))
                wma.io.write_polydata(pd_c, fname_c)
                continue

            add_arrays_c = {}
            add_arrays_c["ClusterIndex"] = c_idx + 1
            add_arrays_c["ClusterUratio"] = add_arrays["ClusterUratio"][c_idx]
            add_arrays_c["ClusterNoF"] = add_arrays["ClusterNoF"][c_idx]
            add_arrays_c["ClusterProb"] = add_arrays["ClusterProb"][c_idx]
            add_arrays_c["ClusterDB"] = add_arrays["ClusterDB"][c_idx]
            add_arrays_c["ClusterAlpha"] = add_arrays["ClusterAlpha"][c_idx]
            add_arrays_c["ClusterTSPC"] = add_arrays["ClusterTSPC"][c_idx]
            add_arrays_c["ClusterWMPG"] = add_arrays["ClusterWMPG"][c_idx]

            # add_arrays_c["SubClusterNoF"] = add_arrays["SubClusterNoF"][c_idx]
            # add_arrays_c["SubClusterUratio"] = add_arrays["SubClusterUratio"][c_idx]
            # add_arrays_c["SubClusterProb"] = add_arrays["SubClusterProb"][c_idx]
            # add_arrays_c["SubClusterDB"] = add_arrays["SubClusterDB"][c_idx]
            # add_arrays_c["SubClusterAlpha"] = add_arrays["SubClusterAlpha"][c_idx]
            # add_arrays_c["SubClusterTSPC"] = add_arrays["SubClusterTSPC"][c_idx]

            for key, value in add_arrays_c.items():
                pd_c = add_vtk_arry(pd_c, key, value)

            appender.AddInputData(pd_c)

        else:
            fname_c = os.path.join(outlier_folder, 'cluster_{0:05d}.vtp'.format(c_idx + 1 - num_clusters))
            wma.io.write_polydata(pd_c, fname_c)

    appender.Update()
    pd_clusters_appended = appender.GetOutput()

    ouput_appedned_folder = os.path.join(cluster_folder, 'appendedvtp')
    os.makedirs(ouput_appedned_folder, exist_ok=True)
    wma.io.write_polydata(pd_clusters_appended, os.path.join(ouput_appedned_folder, 'SWM_all_clusters.vtp'))
    print_both(params["log"], "Save output clusters at:\t %s" % ouput_appedned_folder)

    # Add this new code after the existing cluster output logic
    if params["output_clusters"]:
        # Create a directory for subject-specific clusters
        subject_cluster_folder = os.path.join(output_dir, 'subject_clusters')
        os.makedirs(subject_cluster_folder, exist_ok=True)

        # Get unique subject IDs
        unique_subject_ids = numpy.unique(fiber_subID)

        # Generate distinct colors for the clusters
        num_clusters = params.get('num_clusters', len(numpy.unique(preds_final)) - 1)
        cluster_colors = generate_distinct_colors(num_clusters)

        for subject_id in unique_subject_ids:
            subject_folder = os.path.join(subject_cluster_folder, f'subject_{subject_id}')
            os.makedirs(subject_folder, exist_ok=True)

            # Create subdirectories for each subject
            clusters_folder = os.path.join(subject_folder, 'clusters')
            removed_clusters_folder = os.path.join(subject_folder, 'removed_clusters')
            outliers_folder = os.path.join(subject_folder, 'outliers')
            other_data_folder = os.path.join(subject_folder, 'other_data')

            os.makedirs(clusters_folder, exist_ok=True)
            os.makedirs(removed_clusters_folder, exist_ok=True)
            os.makedirs(outliers_folder, exist_ok=True)
            os.makedirs(other_data_folder, exist_ok=True)

            # Create a mask for the current subject
            subject_mask = fiber_subID == subject_id

            # Create subject-specific predictions and probabilities
            subject_preds = preds_final[subject_mask]
            subject_probs = probs_final[subject_mask]

            # Create a new polydata for the subject
            subject_pd = vtk.vtkPolyData()
            subject_pd.DeepCopy(input_pd)
            
            # Remove fibers that don't belong to this subject
            ids = vtk.vtkIdTypeArray()
            for i in range(input_pd.GetNumberOfCells()):
                if subject_mask[i]:
                    ids.InsertNextValue(i)
            
            selection = vtk.vtkSelectionNode()
            selection.SetFieldType(vtk.vtkSelectionNode.CELL)
            selection.SetContentType(vtk.vtkSelectionNode.INDICES)
            selection.SetSelectionList(ids)

            # Create a vtkSelection and add the selection node to it
            selection_object = vtk.vtkSelection()
            selection_object.AddNode(selection)

            extract = vtk.vtkExtractSelection()
            extract.SetInputData(0, subject_pd)
            extract.SetInputData(1, selection_object)
            extract.Update()

            # Convert the output back to vtkPolyData
            geometry_filter = vtk.vtkGeometryFilter()
            geometry_filter.SetInputData(extract.GetOutput())
            geometry_filter.Update()
            subject_pd = geometry_filter.GetOutput()

            # Output clusters for this subject
            pd_c_list = wma.cluster.mask_all_clusters(subject_pd, subject_preds, num_clusters*2, preserve_point_data=True, preserve_cell_data=True, verbose=False)

            appender = vtk.vtkAppendPolyData()

            for c_idx in range(num_clusters*2):
                pd_c = pd_c_list[c_idx]
                if c_idx < num_clusters:
                    # pd_c = add_vtk_arry(pd_c, "FiberProb", subject_probs[subject_preds == c_idx])
                    
                    # Add solid color information
                    color = cluster_colors[c_idx]
                    color_array = vtk.vtkUnsignedCharArray()
                    color_array.SetNumberOfComponents(3)
                    color_array.SetName("Colors")
                    for _ in range(pd_c.GetNumberOfPoints()):
                        color_array.InsertNextTuple3(int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    pd_c.GetPointData().SetScalars(color_array)

                    if freorder:
                        reorder = cluster_reordered_fibers[c_idx]
                        pd_c = fiber_reorder(pd_c, reorder)

                    if c_idx not in clusters_to_remove:
                        fname_c = os.path.join(clusters_folder, f'cluster_{c_idx+1:05d}.vtp')
                    else:
                        fname_c = os.path.join(removed_clusters_folder, f'cluster_{c_idx+1:05d}.vtp')
                    
                    wma.io.write_polydata(pd_c, fname_c)

                    # add_arrays_c = {
                    #     "ClusterIndex": c_idx + 1,
                    #     "ClusterUratio": add_arrays["ClusterUratio"][c_idx],
                    #     "ClusterNoF": add_arrays["ClusterNoF"][c_idx],
                    #     "ClusterProb": add_arrays["ClusterProb"][c_idx],
                    #     "ClusterDB": add_arrays["ClusterDB"][c_idx],
                    #     "ClusterAlpha": add_arrays["ClusterAlpha"][c_idx],
                    #     "ClusterTSPC": add_arrays["ClusterTSPC"][c_idx],
                    #     "ClusterWMPG": add_arrays["ClusterWMPG"][c_idx],
                    # }

                    # for key, value in add_arrays_c.items():
                    #     pd_c = add_vtk_arry(pd_c, key, value)

                    # appender.AddInputData(pd_c)

                else:
                    # For outliers, use black color
                    outlier_color = (0, 0, 0)  # Black color
                    color_array = vtk.vtkUnsignedCharArray()
                    color_array.SetNumberOfComponents(3)
                    color_array.SetName("Colors")
                    for _ in range(pd_c.GetNumberOfPoints()):
                        color_array.InsertNextTuple3(int(outlier_color[0]*255), int(outlier_color[1]*255), int(outlier_color[2]*255))
                    pd_c.GetPointData().SetScalars(color_array)

                    fname_c = os.path.join(outliers_folder, f'outlier_cluster_{c_idx+1-num_clusters:05d}.vtp')
                    wma.io.write_polydata(pd_c, fname_c)

            # appender.Update()
            # pd_clusters_appended = appender.GetOutput()

            # wma.io.write_polydata(pd_clusters_appended, os.path.join(other_data_folder, 'all_clusters.vtp'))
            print_both(params["log"], f"Saved subject {subject_id} data at: {subject_folder}")

    return pd_clusters_appended