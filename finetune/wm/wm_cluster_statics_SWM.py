#!/usr/bin/env python
import glob
import os
import argparse
import multiprocessing
import numpy
import vtk
import nibabel
from nibabel.affines import apply_affine
#import cifti
import copy

try:
    import whitematteranalysis as wma
except:
    print("<wm_laterality.py> Error importing white matter analysis package\n")
    raise

def compute_lengths(inpd):
    """Compute length of each fiber in polydata. Returns lengths and step size.
    Step size is estimated using points in the middle of a fiber with over 15 points.
    """

    # Make sure we have lines and points.
    if (inpd.GetNumberOfLines() == 0) or (inpd.GetNumberOfPoints() == 0):
        print("<filter.py> No fibers found in input polydata.")
        return 0, 0
    
    # measure step size (using first line that has >=5 points)
    cell_idx = 0
    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()
    inpd.GetLines().InitTraversal()
    while (ptids.GetNumberOfIds() < 5) & (cell_idx < inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)
        ##    inpd.GetLines().GetCell(cell_idx, ptids)
        ## the GetCell function is not wrapped in Canopy python-vtk
        cell_idx += 1
    # make sure we have some points along this fiber
    # In case all fibers in the brain are really short, treat it the same as no fibers.
    if ptids.GetNumberOfIds() < 5:
        return 0, 0
    
    # Use points from the middle of the fiber to estimate step length.
    # This is because the step size may vary near endpoints (in order to include
    # endpoints when downsampling the fiber to reduce file size).
    step_size = 0.0
    count = 0.0
    for ptidx in range(1, ptids.GetNumberOfIds()-1):
        point0 = inpoints.GetPoint(ptids.GetId(ptidx))
        point1 = inpoints.GetPoint(ptids.GetId(ptidx + 1))
        step_size += numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(point0, point1), 2)))
        count += 1
    step_size = step_size / count

    fiber_lengths = list()
    # loop over lines
    inpd.GetLines().InitTraversal()
    num_lines = inpd.GetNumberOfLines()
    for lidx in range(0, num_lines):
        inpd.GetLines().GetNextCell(ptids)
        # save length
        fiber_lengths.append(ptids.GetNumberOfIds() * step_size)

    return numpy.array(fiber_lengths), step_size


def preprocess(inpd):
    """Remove fibers below a length threshold and using other criteria (optional).
    Based on fiber length, and optionally on distance between
    endpoints (u-shape has low distance), and inferior location
    (likely in brainstem).
    """

    fiber_lengths, step_size = compute_lengths(inpd)
    # print(fiber_lengths, step_size)

    # set up processing and output objects
    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()
    ednpoint_dists = [];

    # loop over lines
    inpd.GetLines().InitTraversal()
    num_lines = inpd.GetNumberOfLines()

    for lidx in range(0, num_lines):
        inpd.GetLines().GetNextCell(ptids)

        ptid = ptids.GetId(0)
        point0 = inpoints.GetPoint(ptid)
        ptid = ptids.GetId(ptids.GetNumberOfIds() - 1)
        point1 = inpoints.GetPoint(ptid)

        endpoint_dist = numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(point0, point1), 2)))
        ednpoint_dists.append(endpoint_dist)

    return (fiber_lengths, numpy.array(ednpoint_dists))



def cluster_stat(inpd):

    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(inpd, points_per_fiber=8)
    fiber_array.hemisphere_percent_threshold = 0.5001
    fiber_array.calculate_hemispheres()

    if fiber_array.index_commissure.shape[0] > 0:
        print("There should be no commissural fibers!!!!")

    hemi_mask = numpy.zeros(inpd.GetNumberOfLines())
    hemi_mask[fiber_array.index_right_hem] = 1
    if fiber_array.index_commissure.shape[0] > 0:
        hemi_mask[fiber_array.index_commissure] = 1

    pd_hemis = wma.cluster.mask_all_clusters(inpd, hemi_mask, 2, preserve_point_data=True, preserve_cell_data=True, verbose=False)

    return_str_both = ""
    for pd_hemi in pd_hemis:

        if pd_hemi.GetNumberOfLines() == 0:
            return_str = str(0) + ',' + str(0) +',' + str(0) + ',' + str(0) + ',' + ','
            return_str_both += return_str + ','
            continue

        inpoints = pd_hemi.GetPoints()
        inpointdata = pd_hemi.GetPointData()

        mask = numpy.zeros(pd_hemi.GetNumberOfLines())

        SUB_arrary = inpointdata.GetArray("cluster_idx")
        END_arrary = inpointdata.GetArray("end_label")

        fiber_array = wma.fibers.FiberArray()
        fiber_array.convert_from_polydata(pd_hemi, points_per_fiber=8)
        first_fiber = fiber_array.get_fiber(0)

        pd_hemi.GetLines().InitTraversal()

        end_region_1 = []
        end_region_2 = []
        sub_indice = []
        for lidx in range(0, pd_hemi.GetNumberOfLines()):

            ptids = vtk.vtkIdList()
            pd_hemi.GetLines().GetNextCell(ptids)
            
            line_sub_indice = []
            for pidx in range(0, ptids.GetNumberOfIds()):
                point = inpoints.GetPoint(ptids.GetId(pidx))
                line_sub_indice.append(SUB_arrary.GetTuple(ptids.GetId(pidx))[0])

            sub_indice.append(numpy.unique(line_sub_indice)[0])

            fiber_curr = fiber_array.get_fiber(lidx)
            fiber_curr_match = first_fiber.match_order(fiber_curr)

            reverse = False
            if fiber_curr.r[0] == fiber_curr_match.r[0]:
                reverse = True
                pidx_1 = 0
                pidx_2 = ptids.GetNumberOfIds()-1
            else: 
                pidx_1 = ptids.GetNumberOfIds()-1
                pidx_2 = 0

            end_region_1.append(END_arrary.GetTuple(ptids.GetId(pidx_1))[0])
            end_region_2.append(END_arrary.GetTuple(ptids.GetId(pidx_2))[0])

        sub_indice = numpy.array(sub_indice)

        NoFiber = pd_hemi.GetNumberOfLines()
        NoSubject = numpy.unique(sub_indice).shape[0]

        endregion = ""

        values_1, counts_1 = numpy.unique(end_region_1, return_counts=True)
        arg = numpy.argsort(counts_1)
        arg = numpy.flip(arg)
        values_1 = values_1[arg]
        counts_1 = counts_1[arg]

        values_2, counts_2 = numpy.unique(end_region_2, return_counts=True)
        arg = numpy.argsort(counts_2)
        arg = numpy.flip(arg)
        values_2 = values_2[arg]
        counts_2 = counts_2[arg]

        for v_1, c_1 in zip(values_1.astype(int), counts_1):
            if c_1 / NoFiber < 0.1:
                break
            endregion += "%d(%.3f) - " % (v_1, c_1 / NoFiber)
        endregion = endregion[:-3]
        endregion += ","

        for v_2, c_2 in zip(values_2.astype(int), counts_2):
            if c_2 / NoFiber < 0.1:
                break
            endregion += "%d(%.3f) - " % (v_2, c_2 / NoFiber)
        endregion = endregion[:-3]

        fiber_lengths, ednpoint_dists = preprocess(pd_hemi)

        U_ratio = ednpoint_dists / fiber_lengths

        U_ratio_mean = numpy.mean(U_ratio)
        fiber_len = numpy.mean(fiber_lengths)

        return_str = str(fiber_len) + ',' + str(U_ratio_mean) +',' + str(NoFiber) + ',' + str(NoSubject) + ',' + endregion
        return_str_both += return_str + ','
    
    return_str_both = return_str_both[:-1]
    print(return_str_both)
    return return_str_both


def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Applies preprocessing to input directory. Downsamples, removes short fibers. Preserves tensors and scalar point data along retained fibers.",
        epilog="Written by Lauren O\'Donnell, odonnell@bwh.harvard.edu")
    
    parser.add_argument(
        'inputdir',
        help='')

    parser.add_argument(
        'outputFile',
        help='The output directory should be a new empty directory. It will be created if needed.')


    args = parser.parse_args()
    
    print("Input File", args.inputdir)
    print("Output File", args.outputFile)
    
    print("")
    print("=====input directory======\n", args.inputdir)
    print("=====output directory=====\n", args.outputFile)
    print("==========================")

    
    # =======================================================================
    # Above this line is argument parsing. Below this line is the pipeline.
    # =======================================================================
    def list_files(input_dir,str):
        # Find input files
        input_mask = ("{0}/"+str+"*.vtp").format(input_dir)
        input_pd_fnames = glob.glob(input_mask)
        input_pd_fnames = sorted(input_pd_fnames)
        return(input_pd_fnames)


    vtkfiles = list_files(args.inputdir, 'cluster')

    str_ = "ClusterName,L.Len,L.UR,L.NoF,L.NoS,L.Ep1,L.Ep2,R.Len,R.UR,R.NoF,R.NoS,R.Ep1,R.Ep2\n"
    for vtkfile in vtkfiles:
        print("### cluster:", vtkfile)
        pd = wma.io.read_polydata(vtkfile)
        stat = cluster_stat(pd)
        str_ += os.path.basename(vtkfile) + "," + stat + "\n"

    print(str_)

    data_qc_file = open(args.outputFile, 'w')
    data_qc_file.write(str_)
    data_qc_file.close()

    print('Save to:', args.outputFile)

if __name__ == '__main__':
    main()
