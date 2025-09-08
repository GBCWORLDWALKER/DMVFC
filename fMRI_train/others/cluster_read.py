# find clusters clusters that connect precentral and postcentral ciortical regions
from __future__ import print_function, division
import numpy
import whitematteranalysis as wma
import utils.fibers as fibers
import glob
import argparse
import os


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
        '-indir',action="store", dest="inputDirectory",default="/home/annabelchen/PycharmProjects/SWMA/results_roi",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-outfolder',action="store", dest="outputfolder",default="prg_pog",
        help='Output folder of selected clsuters.')
    parser.add_argument(
        '-thr', action="store", dest="thr", type=float, default=0.1,
        help='percentage threshold for selecting clusters.')
    parser.add_argument('-roi_labels', default=[165,171,166,172],nargs='+',type=int, 
    help='regions for selecting clusters: [left1 left2 right1 right2]')
    parser.add_argument(
        '-trf', action="store", dest="numberOfFibers_train", type=int, default=None,
        help='Number of fibers of each training data to analyze from each subject.')
    parser.add_argument(
        '-l', action="store", dest="fiberLength", type=int, default=40,
        help='Minimum length (in mm) of fibers to analyze. 60mm is default.')
    parser.add_argument(
        '-p', action="store", dest="numberOfFiberPoints", type=int, default=14,
        help='Number of points in each fiber to process. 10 is default.')
    args = parser.parse_args()
    print(args)

    data_dir = args.inputDirectory
    thr = args.thr
    out_dir = data_dir+'/' + args.outputfolder+'_{}'.format(thr)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    labels=args.roi_labels
    input_pd_fnames = wma.io.list_vtk_files(data_dir)
    num_pd = len(input_pd_fnames)
    input_pds = []
    x_arrays = []
    d_rois = []
    fiber_surfs_dk = []
    surf_uniqus=[]
    id_lefts=[]
    id_rights = []
    for i in range(num_pd):
        input_pd, x_array, d_roi, fiber_surf_dk = \
            convert_fiber_to_array(input_pd_fnames[i], numberOfFibers=args.numberOfFibers_train,
                                   fiberLength=args.fiberLength,
                                   numberOfFiberPoints=args.numberOfFiberPoints, preproces=False)
        # surf_unique,surf_counts=numpy.unique(fiber_surf_dk,return_counts=True)
        # surf_uniqus.append(numpy.array(surf_unique))
        # surf_rois=[]
        # for j,count in enumerate(list(surf_counts)):
        #     if count> len(fiber_surf_dk)*0.2:
        #         surf_roi=surf_unique[j]
        #         surf_rois.append(surf_roi)
        # surf_uniqus.append(numpy.array(surf_rois))
        mask1=numpy.all(fiber_surf_dk==[labels[0],labels[1]],axis=1)
        mask2 = numpy.all(fiber_surf_dk == [labels[1], labels[0]], axis=1)
        maskl=mask1+mask2
        per_fiber_left=numpy.sum(maskl)/fiber_surf_dk.shape[0]
        if per_fiber_left>thr:
            id_lefts.append(i+1)

        mask1=numpy.all(fiber_surf_dk==[labels[2],labels[3]],axis=1)
        mask2 = numpy.all(fiber_surf_dk == [labels[3], labels[2]], axis=1)
        maskr=mask1+mask2
        per_fiber_right=numpy.sum(maskr)/fiber_surf_dk.shape[0]
        if per_fiber_right>thr:
            id_rights.append(i+1)

    print(id_lefts)
    print(id_rights)
    fnamesl=[]
    fnamesr = []
    for id_left in id_lefts:
        fnamel='cluster_{}'.format(str(id_left).zfill(5))+'.vtp'
        fnamesl.append(fnamel)
        cmd_cp='cp -r '+data_dir+'/cluster_{}'.format(str(id_left).zfill(5))+'.vtp '+out_dir+'/cluster_{}'.format(str(id_left).zfill(5))+'.vtp'
        os.system(cmd_cp)
    for id_right in id_rights:
        fnamer='cluster_{}'.format(str(id_right).zfill(5))+'.vtp'
        fnamesr.append(fnamer)
        cmd_cp='cp -r '+data_dir+'/cluster_{}'.format(str(id_right).zfill(5))+'.vtp '+out_dir+'/cluster_{}'.format(str(id_right).zfill(5))+'.vtp'
        os.system(cmd_cp)

    cluster_colors = numpy.random.randint(0, 255, (len(fnamesl), 3))
    fname = os.path.join(out_dir, 'clustered_tracts_display_100_percent_left.mrml')
    wma.mrml.write(fnamesl, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)
    cluster_colors = numpy.random.randint(0, 255, (len(fnamesr), 3))
    fname = os.path.join(out_dir, 'clustered_tracts_display_100_percent_right.mrml')
    wma.mrml.write(fnamesr, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)
    
    # clu_left=[]
    # clu_right=[]
    # for i,surf_unique in enumerate(surf_uniqus):
    #     if 165 in surf_unique and 171 in surf_unique:
    #         clu_left.append(i+1)
    #     if 166 in surf_unique and 172 in surf_unique:
    #         clu_right.append(i+1)
    # print(clu_left)
    # print(clu_right)







