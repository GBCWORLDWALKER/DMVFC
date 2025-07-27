import argparse
import numpy as np
import os
import whitematteranalysis as wma
import numpy
import glob
import vtk

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

def get_centroid(feat,number_fibers_in_subject_cluster):
    if number_fibers_in_subject_cluster==1:
        centroid=feat[0]
        return centroid
    # sample_number = 100
    # if number_fibers_in_subject_cluster > 1000:
    #     index = numpy.random.randint(0, number_fibers_in_subject_cluster, sample_number)
    #     feat = feat[index]

    distance_array = numpy.zeros((len(feat), len(feat)))
    distance_sum = numpy.zeros((len(feat)))
    for j in range(len(feat)):
        #print(j)
        # if j==99:
        #     print('debug')
        fiber = feat[j]
        distance = fiber_distance.fiber_distance(fiber, feat)
        distance_array[j, :] = distance
        distance_sum[j] = numpy.sum(distance)
    centroid=feat[numpy.argmin(distance_sum)]
    return centroid


def main():
    parser = argparse.ArgumentParser(
        description="Calculate evalution metrics for all subjects")
    parser.add_argument('-sub_dir', dest='subject_dir', default='../dataFolder/cluster_dcec/angle_distance/final',
                        help='A directory containing cluster centroids of test subjects.')
    parser.add_argument('-out_dir', dest='out_dir', default='../dataFolder/cluster_dcec/mean_distance/final',
                        help='A directory containing cluster centroids of the refrence subject.')

    args = parser.parse_args()
    print('sub_dir',args.subject_dir)
    print('out_dir', args.out_dir)
    
    input_mask = "{0}/cluster_*.vtp".format(os.path.join(args.subject_dir))
    subject_clusters = sorted(glob.glob(input_mask))
    
    U_ratios = []
    str_all = ""
    for c, cs in enumerate(subject_clusters):
        print(cs)

        inpd = wma.io.read_polydata(cs)

        fiber_lengths, ednpoint_dists = preprocess(inpd)

        U_ratio = ednpoint_dists / fiber_lengths

        U_ratio_mean = numpy.mean(U_ratio)

        fiber_len = numpy.mean(fiber_lengths)

        str_='%d,%f,%f'%(c+1, U_ratio_mean, fiber_len)
        print(str_)

        str_all += str_ + '\n'

        U_ratios.append(U_ratio_mean)

    U_ratios = numpy.array(U_ratios)
    cluster_ids = np.argsort(U_ratios)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for c_idx, cluster_id in enumerate(cluster_ids):
        os.system("cp %s/cluster_%05d.vtp %s/cluster_%05d.vtp" % (args.subject_dir, cluster_id+1, args.out_dir, c_idx+1))

    os.system("cp %s/*.mrml %s/" % (args.subject_dir, args.out_dir))


if __name__ == '__main__':
    main()
