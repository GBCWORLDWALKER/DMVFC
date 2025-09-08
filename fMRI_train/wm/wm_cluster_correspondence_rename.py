import argparse
import numpy as np
import os
import numpy
import glob
import sys

import whitematteranalysis as wma

sys.path.insert(0, '/home/fan/Projects/SWM_DFC/')
from utils import fibers
from utils import fiber_distance

def main():
    parser = argparse.ArgumentParser(
        description="Calculate evalution metrics for all subjects")
    parser.add_argument('-sub_dir', dest='subject_dir', default='../dataFolder/cluster_dcec/angle_distance/final',
                        help='A directory containing cluster centroids of test subjects.')
    parser.add_argument('-atl_dir', dest='atlas_dir', default='../dataFolder/cluster_dcec/mean_distance/final',
                        help='A directory containing cluster centroids of the refrence subject.')
    parser.add_argument('-out_dir', dest='out_dir', default='../dataFolder/cluster_dcec/mean_distance/final',
                        help='A directory containing cluster centroids of the refrence subject.')

    def get_centroid(feat, number_fibers_in_subject_cluster):
        if number_fibers_in_subject_cluster == 1:
            centroid = feat[0]
            return centroid
        # sample_number = 100
        # if number_fibers_in_subject_cluster > 1000:
        #     index = numpy.random.randint(0, number_fibers_in_subject_cluster, sample_number)
        #     feat = feat[index]

        distance_array = numpy.zeros((len(feat), len(feat)))
        distance_sum = numpy.zeros((len(feat)))
        for j in range(len(feat)):
            # print(j)
            # if j==99:
            #     print('debug')
            fiber = feat[j]
            distance = fiber_distance.fiber_distance(fiber, feat)
            distance_array[j, :] = distance
            distance_sum[j] = numpy.sum(distance)
        centroid = feat[numpy.argmin(distance_sum)]
        return centroid
    args = parser.parse_args()
    print('sub_dir',args.subject_dir)
    print('atlas_dir', args.atlas_dir)

    input_mask = "{0}/cluster_*.vtp".format(os.path.join(args.atlas_dir))
    subject_clusters = sorted(glob.glob(input_mask))
    
    centroids_atl=[]
    num_rois=[]
    for c, cs in enumerate(subject_clusters):
        if c % 50==0:
            print(cs)
        pd_subject = wma.io.read_polydata(cs)
        fiber_array = fibers.FiberArray()
        fiber_array.convert_from_polydata(pd_subject, points_per_fiber=14)
        feat_surf_dk = fiber_array.fiber_surface_dk
        roi,counts=numpy.unique(feat_surf_dk,return_counts=True)
        roi_selected=roi[counts>len(feat_surf_dk)*0.1]
        num_roi=len(roi_selected)
        num_rois.append(num_roi)
        feat_sub = np.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a, fiber_array.fiber_array_s))
        number_subject_cluster = pd_subject.GetNumberOfLines()
        #print('numer:',number_subject_cluster )
        if number_subject_cluster==0:
            centroid_sub=np.ones([14,3])*1000
        else:
            centroid_sub = get_centroid(feat_sub, number_subject_cluster)
        centroids_atl.append(centroid_sub)
    centroids_atl=np.array(centroids_atl)

    
    input_mask = "{0}/cluster_*.vtp".format(os.path.join(args.subject_dir))
    subject_clusters = sorted(glob.glob(input_mask))
    
    centroids_sub=[]
    num_rois=[]
    for c, cs in enumerate(subject_clusters):
        if c % 50 == 0:
            print(cs)
        pd_subject = wma.io.read_polydata(cs)
        fiber_array = fibers.FiberArray()
        fiber_array.convert_from_polydata(pd_subject, points_per_fiber=14)
        feat_surf_dk = fiber_array.fiber_surface_dk
        roi,counts=numpy.unique(feat_surf_dk,return_counts=True)
        roi_selected=roi[counts>len(feat_surf_dk)*0.1]
        num_roi=len(roi_selected)
        num_rois.append(num_roi)
        feat_sub = np.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a, fiber_array.fiber_array_s))
        number_subject_cluster = pd_subject.GetNumberOfLines()
        #print('numer:',number_subject_cluster )
        if number_subject_cluster==0:
            centroid_sub=np.ones([14,3])*1000
        else:
            centroid_sub = get_centroid(feat_sub, number_subject_cluster)
        centroids_sub.append(centroid_sub)
    centroids_sub=np.array(centroids_sub)


    mapped_clusters = []
    for c_idx in range(centroids_atl.shape[0]):

        print("Atlas cluster %05d"% (c_idx+1))

        centroids_atl_cluster = numpy.squeeze(centroids_atl[c_idx, :])

        distance = fiber_distance.fiber_distance(centroids_atl_cluster, centroids_sub)
        cluster_ids = np.argsort(distance)
        cluster_id = np.argmin(distance)


        if cluster_id not in mapped_clusters:
            mapped_clusters.append(cluster_id)
        else:
            while cluster_id in mapped_clusters:
                # print(" # subject cluster %05d mapped alreay, next one is:" % (cluster_id+1))
                distance[cluster_id] = 10000
                cluster_ids = np.argsort(distance)
                cluster_id = np.argmin(distance)
            mapped_clusters.append(cluster_id)
        
        # print(" * mapping to subject cluster %05d" % (cluster_id + 1))

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for c_idx in range(centroids_atl.shape[0]):
        os.system("cp %s/cluster_%05d.vtp %s/cluster_%05d.vtp" % (args.subject_dir, mapped_clusters[c_idx]+1, args.out_dir, c_idx+1))

    os.system("cp %s/*.mrml %s/" % (args.subject_dir, args.out_dir))

if __name__ == '__main__':
    main()
