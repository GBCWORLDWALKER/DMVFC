import argparse
import numpy as np
from utils import fiber_distance
import os
import whitematteranalysis as wma
import numpy
import glob
from utils import fibers

def main():
    parser = argparse.ArgumentParser(
        description="Calculate evalution metrics for all subjects")
    # parser.add_argument('-sub_dir', dest='subject_dir', default='../dataFolder/cluster_dcec/angle_distance/final',
    #                     help='A directory containing cluster centroids of test subjects.')
    # parser.add_argument('-atl_dir', dest='atlas_dir', default='../dataFolder/cluster_dcec/mean_distance/final',
    #                     help='A directory containing cluster centroids of the refrence subject.')

    parser.add_argument('-sub_dir',dest='subject_dir',default='/media/annabelchen/DataShare/deepClustering/journal_data/DFC_subject/HCP/DFCv2',#'/media/annabelchen/DataShare/deepClustering/cluster_dcec/journal_test/PPMI/3551',
                        #'/home/annabelchen/PycharmProjects/torch_DFC/results/QB/101006',#,
                        help='A directory containing cluster centroids of test subjects.')
    parser.add_argument('-atl_dir',dest='atlas_dir',default='/media/annabelchen/DataShare/deepClustering/journal_data/DFC_subject/HCP/DFCv2',#'/home/annabelchen/PycharmProjects/whitematteranalysis/ORG-800FC-100HCP',
                        help='A directory containing cluster centroids of the refrence subject.')
    parser.add_argument('-id',dest='id',default=112,
                        help='A directory containing cluster centroids of the refrence subject.')

    args = parser.parse_args()
    print('sub_dir',args.subject_dir)
    print('atlas_dir', args.atlas_dir)
    print('atlas_id', args.id)
    altas_cluster_id = args.id
    atlas_cluster_name=os.path.join(args.atlas_dir,'cluster_00'+str(altas_cluster_id).zfill(3)+'.vtp') #
    pd_atlas = wma.io.read_polydata(atlas_cluster_name)
    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_atlas, points_per_fiber=14)
    feat = np.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    number_atlas_cluster=pd_atlas.GetNumberOfLines()
    print('number_atlas_cluster',number_atlas_cluster)
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
    centroid_atlas=get_centroid(feat,number_atlas_cluster)

    input_mask = "{0}/cluster_*.vtp".format(os.path.join(args.subject_dir))
    subject_clusters = sorted(glob.glob(input_mask))
    centroids_sub=[]
    num_rois=[]
    for c, cs in enumerate(subject_clusters):
        if c%100==0:
            print(c)
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
    distance = fiber_distance.fiber_distance(centroid_atlas, centroids_sub)
    cluster_ids=np.argsort(distance)
    cluster_id = np.argmin(distance)
    print(cluster_ids[0:8]+1)
    num_rois=numpy.array(num_rois)
    index=numpy.where(num_rois==2)[0]
    print(index+1)

    # if 'atlas' in args.subject_dir:
    #     mrml_files = glob.glob(args.subject_dir + '/T*.mrml')
    #     wma.mrml







if __name__ == '__main__':
    main()
