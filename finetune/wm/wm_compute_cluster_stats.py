#!/usr/bin/env python
import glob
import os
import argparse
import numpy
import pandas
import vtk
import pickle
import nibabel
from nibabel.affines import apply_affine
import copy
import matplotlib.pyplot as plt

try:
    import whitematteranalysis as wma
except:
    print("<wm_laterality.py> Error importing white matter analysis package\n")
    raise

def list_files(input_dir,str):
    # Find input files
    input_mask = ("{0}/"+str+"*.vtp").format(input_dir)
    input_pd_fnames = glob.glob(input_mask)
    input_pd_fnames = sorted(input_pd_fnames)
    return(input_pd_fnames)


def get_fiber_SubID(pd_cluster):
    array_name = "cluster_idx"
    inpointdata = pd_cluster.GetPointData()
    point_array = inpointdata.GetArray(array_name)
    array_val = []
    pd_cluster.GetLines().InitTraversal()
    for lidx in range(0, pd_cluster.GetNumberOfLines()):
        ptids = vtk.vtkIdList()
        pd_cluster.GetLines().GetNextCell(ptids)
        array_val.append(point_array.GetTuple(ptids.GetId(0))[0])
    fiberSubID = numpy.array(array_val).astype(int)
    return fiberSubID


def get_fiber_mean_val(pd_cluster, array_name, fiber_SubID=None):
    inpointdata = pd_cluster.GetPointData()
    point_array = inpointdata.GetArray(array_name)
    array_val = []
    pd_cluster.GetLines().InitTraversal()
    for lidx in range(0, pd_cluster.GetNumberOfLines()):
        ptids = vtk.vtkIdList()
        pd_cluster.GetLines().GetNextCell(ptids)
        tmp_val = numpy.zeros(ptids.GetNumberOfIds())
        for pidx in range(0, ptids.GetNumberOfIds()):
            tmp_val[pidx] = point_array.GetTuple(ptids.GetId(pidx))[0]
        tmp_val_mean = numpy.mean(tmp_val)
        array_val.append(tmp_val_mean)
    fiber_mean_val = numpy.array(array_val)

    if fiber_SubID is not None and fiber_SubID.shape[0] != 0:
        subIDs = numpy.unique(fiber_SubID)
        sub_mean_val = numpy.zeros(subIDs.max() + 1) - 1
        for s_idx in subIDs:
            sub_mean_val[s_idx] = fiber_mean_val[fiber_SubID == s_idx].mean()
        sub_mean_val[sub_mean_val == -1] = numpy.nan
    else:
        sub_mean_val = None

    return fiber_mean_val, sub_mean_val

def separate_hemi_clusters(pd_cluster):
    fiber_array = wma.fibers.FiberArray()
    fiber_array.convert_from_polydata(pd_cluster, points_per_fiber=15)

    array_r = fiber_array.fiber_array_r
    array_r = (numpy.sum(array_r > 0, axis=1) > 8)  # R True; L False
    array_r_mask = array_r.astype(int)

    pd_cluster_r = wma.filter.mask(pd_cluster, array_r_mask, preserve_point_data=True, verbose=False)
    pd_cluster_l = wma.filter.mask(pd_cluster, 1 - array_r_mask, preserve_point_data=True, verbose=False)
    return pd_cluster_l, pd_cluster_r

def _calculate_line_indices(input_line_length, output_line_length):

    # this is the increment between output points
    step = (input_line_length - 1.0) / (output_line_length - 1.0)

    # these are the output point indices (0-based)
    ptlist = []
    for ptidx in range(0, output_line_length):
        ptlist.append(ptidx * step)

    return ptlist


def convert_from_polydata(input_vtk_polydata, points_per_fiber=30):

    number_of_fibers = input_vtk_polydata.GetNumberOfLines()

    # allocate array number of lines by line length
    fiber_array_r = numpy.zeros((number_of_fibers, points_per_fiber), dtype=numpy.float16)
    fiber_array_a = numpy.zeros((number_of_fibers, points_per_fiber), dtype=numpy.float16)
    fiber_array_s = numpy.zeros((number_of_fibers, points_per_fiber), dtype=numpy.float16)

    fiber_surf_dis = numpy.zeros((number_of_fibers, points_per_fiber), dtype=numpy.float16)

    # get data from input polydata
    inpoints = input_vtk_polydata.GetPoints()
    inpointdata = input_vtk_polydata.GetPointData()

    surf_dis_array = inpointdata.GetArray("surf_dis")

    input_vtk_polydata.GetLines().InitTraversal()
    for lidx in range(0, number_of_fibers):

        if lidx % 1000 == 0:
            print("Processing fiber %d / %d:" % (lidx, number_of_fibers))

        line_ptids = vtk.vtkIdList()
        input_vtk_polydata.GetLines().GetNextCell(line_ptids)
        num_of_points = line_ptids.GetNumberOfIds()

        for pidx, line_index in enumerate(_calculate_line_indices(num_of_points, points_per_fiber)):
            # do nearest neighbor interpolation: round index
            ptidx = line_ptids.GetId(int(round(line_index)))
            point = list(inpoints.GetPoint(ptidx))
            fiber_array_r[lidx, pidx] = point[0]
            fiber_array_a[lidx, pidx] = point[1]
            fiber_array_s[lidx, pidx] = point[2]

            fiber_surf_dis[lidx, pidx] = surf_dis_array.GetTuple(ptidx)[0]

    return fiber_array_r, fiber_array_a, fiber_array_s, fiber_surf_dis

def _fiber_distance_internal_use(fiber_r, fiber_a, fiber_s, fiber_array, threshold=0, distance_method='Mean'):
    """ Compute the total fiber distance from one fiber to an array of
    many fibers.
    This function does not handle equivalent fiber representations,
    for that use fiber_distance, above.
    """

    fiber_array_r = fiber_array[:, :, 0]
    fiber_array_a = fiber_array[:, :, 1]
    fiber_array_s = fiber_array[:, :, 2]

    # compute the distance from this fiber to the array of other fibers
    ddx = fiber_array_r - fiber_r
    ddy = fiber_array_a - fiber_a
    ddz = fiber_array_s - fiber_s

    dx = numpy.square(ddx)
    dy = numpy.square(ddy)
    dz = numpy.square(ddz)

    # sum dx dx dz at each point on the fiber and sqrt for threshold
    # distance = numpy.sqrt(dx + dy + dz)
    distance = dx + dy + dz

    # threshold if requested
    if threshold:
        # set values less than threshold to 0
        distance = distance - threshold * threshold
        idx = numpy.nonzero(distance < 0)
        distance[idx] = 0

    if distance_method == 'Mean':
        # sum along fiber
        distance = numpy.sum(numpy.sqrt(distance), 1)
        # Remove effect of number of points along fiber (mean)
        npts = float(fiber_array.shape[1])
        #print(npts)
        distance = distance / npts
        # for consistency with other methods we need to square this value
        #distance = numpy.square(distance)
    elif distance_method == 'Hausdorff':
        # take max along fiber
        distance = numpy.max(distance, 1)
    elif distance_method == 'MeanSquared':
        # sum along fiber
        distance = numpy.sum(distance, 1)
        # Remove effect of number of points along fiber (mean)
        npts = len(fiber_r)
        distance = distance / npts
    elif distance_method == 'StrictSimilarity':
        # for use in laterality
        # this is the product of all similarity values along the fiber
        # not truly a distance but it's easiest to compute here in this function
        # where we have all distances along the fiber
        # print "distance range :", numpy.min(distance), numpy.max(distance)
        #distance = distance_to_similarity(distance, sigmasq)
        # print "similarity range :", numpy.min(distance), numpy.max(distance)
        distance = numpy.prod(distance, 1)
        # print "overall similarity range:", numpy.min(distance), numpy.max(distance)
    elif distance_method == 'Mean_shape':

        # sum along fiber
        distance_square = distance
        distance = numpy.sqrt(distance_square)

        d = numpy.sum(distance, 1)
        # Remove effect of number of points along fiber (mean)
        npts = float(fiber_array.points_per_fiber)
        d = numpy.divide(d, npts)
        # for consistency with other methods we need to square this value
        d = numpy.square(d)

        distance_endpoints = (distance[:, 0] + distance[:, npts - 1]) / 2

        for i in numpy.linspace(0, numpy.size(distance, 0) - 1, numpy.size(distance, 0)):
            for j in numpy.linspace(0, numpy.size(distance, 1) - 1, numpy.size(distance, 1)):
                if distance[i, j] == 0:
                    distance[i, j] = 1
        ddx = numpy.divide(ddx, distance)
        ddy = numpy.divide(ddy, distance)
        ddz = numpy.divide(ddz, distance)
        # print ddx*ddx+ddy*ddy+ddz*ddz
        npts = float(fiber_array.points_per_fiber)
        angles = numpy.zeros([(numpy.size(distance)) / npts, npts * (npts + 1) / 2])
        s = 0
        n = numpy.linspace(0, npts - 1, npts)
        for i in n:
            m = numpy.linspace(0, i, i + 1)
            for j in m:
                angles[:, s] = (ddx[:, i] - ddx[:, j]) * (ddx[:, i] - ddx[:, j]) + (ddy[:, i] - ddy[:, j]) * (
                            ddy[:, i] - ddy[:, j]) + (ddz[:, i] - ddz[:, j]) * (ddz[:, i] - ddz[:, j])
                s = s + 1
        angles = (numpy.sqrt(angles)) / 2
        angle = numpy.max(angles, 1)

        distance = 0.5 * d + 0.4 * d / (0.5 + 0.5 * (1 - angle * angle)) + 0.1 * distance_endpoints

    else:
        print("<fibers.py> throwing Exception. Unknown input distance method (typo?):", distance_method)
        raise Exception("unknown distance method")

    return distance

def convert_to_polydata(fiber_array, arrays=None):
    """Convert fiber array to vtkPolyData object."""

    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()

    fiber_array_r = fiber_array[:, :, 0]
    fiber_array_a = fiber_array[:, :, 1]
    fiber_array_s = fiber_array[:, :, 2]
    number_of_fibers = fiber_array.shape[0]
    points_per_fiber = fiber_array.shape[1]
    outlines.InitTraversal()

    for lidx in range(0, number_of_fibers):

        cellptids = vtk.vtkIdList()

        for pidx in range(0, points_per_fiber):
            idx = outpoints.InsertNextPoint(fiber_array_r[lidx, pidx],
                                            fiber_array_a[lidx, pidx],
                                            fiber_array_s[lidx, pidx])

            cellptids.InsertNextId(idx)

        outlines.InsertNextCell(cellptids)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)

    if arrays is not None:
        for key, value in arrays.items():
            value = numpy.array(value)
            if value.shape[1] == 1:
                outpd = add_vtk_arry(outpd, key, value)
            else:
                outpd = add_vtk_point_arry(outpd, key, value)

    return outpd

def add_vtk_arry(inpd, array_name, array_value):
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName(array_name)
    inpd.GetLines().InitTraversal()
    for lidx in range(0, inpd.GetNumberOfLines()):
        ptids = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(ptids)
        if array_value.size == 1:
            val = array_value
        else:
            val = array_value[lidx]
        for pidx in range(0, ptids.GetNumberOfIds()):
            vtk_array.InsertNextTuple1(val)
    inpd.GetPointData().AddArray(vtk_array)
    inpd.GetPointData().Update()

    return inpd

def add_vtk_point_arry(inpd, array_name, array_value):
    vtk_array = vtk.vtkDoubleArray()
    vtk_array.SetName(array_name)
    inpd.GetLines().InitTraversal()
    for lidx in range(0, inpd.GetNumberOfLines()):
        ptids = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(ptids)
        for pidx in range(0, ptids.GetNumberOfIds()):
            val = array_value[lidx, pidx]
            vtk_array.InsertNextTuple1(val)
    inpd.GetPointData().AddArray(vtk_array)
    inpd.GetPointData().Update()

    return inpd


def get_cluster_centroid_and_alphanew(pd_cluster, hemi, c_idx, atlas_centroid_fiber_array, fiber_SubID=None):

    if hemi == "left":
        atlas_cluster_centroid_r = - atlas_centroid_fiber_array.fiber_array_r[c_idx]
        atlas_cluster_centroid_a = atlas_centroid_fiber_array.fiber_array_a[c_idx]
        atlas_cluster_centroid_s = atlas_centroid_fiber_array.fiber_array_s[c_idx]
    else:
        atlas_cluster_centroid_r = atlas_centroid_fiber_array.fiber_array_r[c_idx]
        atlas_cluster_centroid_a = atlas_centroid_fiber_array.fiber_array_a[c_idx]
        atlas_cluster_centroid_s = atlas_centroid_fiber_array.fiber_array_s[c_idx]

    atlas_cluster_centroid_ras = numpy.dstack(
        (atlas_cluster_centroid_r, atlas_cluster_centroid_a, atlas_cluster_centroid_s))
    atlas_cluster_centroid_ras = numpy.squeeze(atlas_cluster_centroid_ras)

    array_r, array_a, array_s, surf_dis = convert_from_polydata(pd_cluster)

    if surf_dis.shape[0] != 0:
        cluster_fiber_array_ras_orig = numpy.dstack((array_r, array_a, array_s))
        cluster_fiber_array_ras_quiv = numpy.flip(cluster_fiber_array_ras_orig, axis=1)

        dis_orig = _fiber_distance_internal_use(atlas_cluster_centroid_ras[:, 0], atlas_cluster_centroid_ras[:, 1],
                                                atlas_cluster_centroid_ras[:, 2], cluster_fiber_array_ras_orig)
        dis_quiv = _fiber_distance_internal_use(atlas_cluster_centroid_ras[:, 0], atlas_cluster_centroid_ras[:, 1],
                                                atlas_cluster_centroid_ras[:, 2], cluster_fiber_array_ras_quiv)

        dis_tmp = numpy.stack((dis_orig, dis_quiv), axis=0)
        dis_min = numpy.min(dis_tmp, axis=0)
        dis_arg = numpy.argmin(dis_tmp, axis=0)

        x_array_orig_ = cluster_fiber_array_ras_orig[numpy.where(dis_arg == 0)]
        x_array_quiv_ = cluster_fiber_array_ras_quiv[numpy.where(dis_arg == 1)]

        x_array_reodered = numpy.concatenate((x_array_orig_, x_array_quiv_))
        x_array_reodered_mean = numpy.mean(x_array_reodered, axis=0)

        # surf distance
        surf_dis_reodered = copy.deepcopy(surf_dis)
        surf_dis_reodered[numpy.where(dis_arg == 1)] = numpy.flip(surf_dis[numpy.where(dis_arg == 1)], axis=1)

        fiber_dis_to_centroid = dis_min  # (number of fiber, )
        fiber_surf_dis = surf_dis_reodered  # (number of fiber, number of points)
        fiber_reorder = dis_arg  # (number of fiber, )

        # return values
        cluster_centroid = x_array_reodered_mean  # (number of points, 3)
        cluster_surf_dis = surf_dis_reodered.mean(axis=0)  # (number of points, )

        cluster_alphanew = numpy.std(fiber_dis_to_centroid) / numpy.mean(fiber_dis_to_centroid)
        if fiber_SubID is not None:
            subIDs = numpy.unique(fiber_SubID)
            sub_alphanew = numpy.zeros(subIDs.max() + 1) - 1
            for s_idx in subIDs:
                sub_val = fiber_dis_to_centroid[fiber_SubID == s_idx]
                sub_alphanew[s_idx] = numpy.std(sub_val) / numpy.mean(sub_val)
            sub_alphanew[sub_alphanew == -1] = numpy.nan
        else:
            sub_alphanew = None

    else:
        cluster_centroid = numpy.zeros((30, 3))
        cluster_surf_dis = numpy.nan
        cluster_alphanew = numpy.nan
        sub_alphanew = numpy.nan
        fiber_surf_dis = numpy.nan
        fiber_reorder = numpy.nan

    return cluster_centroid, - cluster_surf_dis, cluster_alphanew, \
           sub_alphanew, - fiber_surf_dis, fiber_reorder


def main():
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(
        description="Compute additional cluster stats",
        epilog="Written by Fan Zhang")

    parser.add_argument(
        'inputdir',
        help='dir with the input clusters')

    args = parser.parse_args()
    vtkfiles = list_files(args.inputdir, 'cluster')

    atlas_centroid_vtk = glob.glob(os.path.join(args.inputdir, "*centroid.vtp"))[0]
    pd_atlas_centroid = wma.io.read_polydata(atlas_centroid_vtk)
    atlas_centroid_fiber_array = wma.fibers.FiberArray()
    atlas_centroid_fiber_array.convert_from_polydata(pd_atlas_centroid, points_per_fiber=30)

    total_sub = 100

    cluster_WMPG_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_WMPG_LI_list = numpy.zeros((len(vtkfiles)*2, 1))

    cluster_NoS_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_NoS_LI_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_NoS_sub_CV_list = numpy.zeros((len(vtkfiles)*2, 1))

    cluster_FA_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_FA_LI_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_FA_sub_CV_list = numpy.zeros((len(vtkfiles)*2, 1))

    cluster_UR_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_UR_LI_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_UR_sub_CV_list = numpy.zeros((len(vtkfiles)*2, 1))

    cluster_alphanew_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_alphanew_LI_list = numpy.zeros((len(vtkfiles)*2, 1))
    cluster_alphanew_sub_CV_list = numpy.zeros((len(vtkfiles)*2, 1))

    cluster_surf_dis = numpy.zeros((len(vtkfiles)*2, 30))

    cluster_centroids = numpy.zeros((len(vtkfiles)*2, 30, 3))

    for idx, vtkfile in enumerate(vtkfiles[::1]):
        print(vtkfile)

        # if idx != 1385:
        #     continue

        pd_cluster = wma.io.read_polydata(vtkfile)
        pd_cluster_l, pd_cluster_r = separate_hemi_clusters(pd_cluster)

        # WMPG
        fiber_SubID_l = get_fiber_SubID(pd_cluster_l)
        fiber_SubID_r = get_fiber_SubID(pd_cluster_r)
        cluster_WMPG_l = numpy.unique(fiber_SubID_l).shape[0] / total_sub
        cluster_WMPG_r = numpy.unique(fiber_SubID_r).shape[0] / total_sub
        cluster_WMPG_LI = (cluster_WMPG_l - cluster_WMPG_r) / (cluster_WMPG_l + cluster_WMPG_r)

        cluster_WMPG_list[2 * idx] = cluster_WMPG_l
        cluster_WMPG_list[2 * idx + 1] = cluster_WMPG_r
        cluster_WMPG_LI_list[2 * idx] = cluster_WMPG_LI
        cluster_WMPG_LI_list[2 * idx + 1] = cluster_WMPG_LI

        # NoS
        cluster_NoS_l = pd_cluster_l.GetNumberOfLines()
        cluster_NoS_r = pd_cluster_r.GetNumberOfLines()
        cluster_NoS_LI = (cluster_NoS_l - cluster_NoS_r) / (cluster_NoS_l + cluster_NoS_r)

        cluster_sub_NoS_l = numpy.zeros(total_sub)
        cluster_sub_NoS_r = numpy.zeros(total_sub)
        sidx, freq = numpy.unique(fiber_SubID_l, return_counts = True)
        cluster_sub_NoS_l[sidx] = freq
        sidx, freq = numpy.unique(fiber_SubID_r, return_counts = True)
        cluster_sub_NoS_r[sidx] = freq
        cluster_sub_NoS_CV_l = numpy.nanstd(cluster_sub_NoS_l) / numpy.nanmean(cluster_sub_NoS_l)
        cluster_sub_NoS_CV_r = numpy.nanstd(cluster_sub_NoS_r) / numpy.nanmean(cluster_sub_NoS_r)

        cluster_NoS_list[2 * idx] = cluster_NoS_l
        cluster_NoS_list[2 * idx + 1] = cluster_NoS_r
        cluster_NoS_LI_list[2 * idx] = cluster_NoS_LI
        cluster_NoS_LI_list[2 * idx + 1] = cluster_NoS_LI
        cluster_NoS_sub_CV_list[2 * idx] = cluster_sub_NoS_CV_l
        cluster_NoS_sub_CV_list[2 * idx + 1] = cluster_sub_NoS_CV_r

        # FA
        fiber_mean_FA_l, sub_mean_FA_l = get_fiber_mean_val(pd_cluster_l, "FA1", fiber_SubID=fiber_SubID_l)
        fiber_mean_FA_r, sub_mean_FA_r = get_fiber_mean_val(pd_cluster_r, "FA1", fiber_SubID=fiber_SubID_r)

        cluster_mean_FA_l = fiber_mean_FA_l.mean()
        cluster_mean_FA_r = fiber_mean_FA_r.mean()
        cluster_FA_LI = (cluster_mean_FA_l - cluster_mean_FA_r) / (cluster_mean_FA_l + cluster_mean_FA_r)

        cluster_sub_FA_CV_l = numpy.nanstd(sub_mean_FA_l) / numpy.nanmean(sub_mean_FA_l) if not numpy.isnan(cluster_mean_FA_l) else numpy.nan
        cluster_sub_FA_CV_r = numpy.nanstd(sub_mean_FA_r) / numpy.nanmean(sub_mean_FA_r) if not numpy.isnan(cluster_mean_FA_r) else numpy.nan

        cluster_FA_list[2 * idx] = cluster_mean_FA_l
        cluster_FA_list[2 * idx + 1] = cluster_mean_FA_r
        cluster_FA_LI_list[2 * idx] = cluster_FA_LI
        cluster_FA_LI_list[2 * idx + 1] = cluster_FA_LI
        cluster_FA_sub_CV_list[2 * idx] = cluster_sub_FA_CV_l
        cluster_FA_sub_CV_list[2 * idx + 1] = cluster_sub_FA_CV_r


        # U-ratio
        fiber_mean_UR_l, sub_mean_UR_l = get_fiber_mean_val(pd_cluster_l, "Uratio", fiber_SubID=fiber_SubID_l)
        fiber_mean_UR_r, sub_mean_UR_r = get_fiber_mean_val(pd_cluster_r, "Uratio", fiber_SubID=fiber_SubID_r)

        cluster_mean_UR_l = fiber_mean_UR_l.mean()
        cluster_mean_UR_r = fiber_mean_UR_r.mean()
        cluster_UR_LI = (cluster_mean_UR_l - cluster_mean_UR_r) / (cluster_mean_UR_l + cluster_mean_UR_r)

        cluster_sub_UR_CV_l = numpy.nanstd(sub_mean_UR_l) / numpy.nanmean(sub_mean_UR_l) if not numpy.isnan(cluster_mean_UR_l) else numpy.nan
        cluster_sub_UR_CV_r = numpy.nanstd(sub_mean_UR_r) / numpy.nanmean(sub_mean_UR_r) if not numpy.isnan(cluster_mean_UR_r) else numpy.nan

        cluster_UR_list[2 * idx] = cluster_mean_UR_l
        cluster_UR_list[2 * idx + 1] = cluster_mean_UR_r
        cluster_UR_LI_list[2 * idx] = cluster_UR_LI
        cluster_UR_LI_list[2 * idx + 1] = cluster_UR_LI
        cluster_UR_sub_CV_list[2 * idx] = cluster_sub_UR_CV_l
        cluster_UR_sub_CV_list[2 * idx + 1] = cluster_sub_UR_CV_r


        # Alpha and Surfdis
        cluster_centroid_l, cluster_surf_dis_l, \
        cluster_alphanew_l, sub_alphanew_l, \
        fiber_surf_dis_l, fiber_reorder_l = \
            get_cluster_centroid_and_alphanew(pd_cluster_l, "left", idx, atlas_centroid_fiber_array, fiber_SubID=fiber_SubID_l)

        cluster_centroid_r, cluster_surf_dis_r, \
        cluster_alphanew_r, sub_alphanew_r, \
        fiber_surf_dis_r, fiber_reorder_r = \
            get_cluster_centroid_and_alphanew(pd_cluster_r, "right", idx, atlas_centroid_fiber_array, fiber_SubID=fiber_SubID_r)
        # TODO: not sure if fiber_surf_dis_r is useful at all.

        cluster_alphanew_LI = (cluster_alphanew_l - cluster_alphanew_r) / (cluster_alphanew_l + cluster_alphanew_r)
        cluster_alphanew_sub_CV_l = numpy.nanstd(sub_alphanew_l) / numpy.nanmean(sub_alphanew_l)
        cluster_alphanew_sub_CV_r = numpy.nanstd(sub_alphanew_r) / numpy.nanmean(sub_alphanew_r)

        cluster_alphanew_list[idx * 2] = cluster_alphanew_l
        cluster_alphanew_list[idx * 2 + 1] = cluster_alphanew_r
        cluster_alphanew_LI_list[idx * 2] = cluster_alphanew_LI
        cluster_alphanew_LI_list[idx * 2 + 1] = cluster_alphanew_LI
        cluster_alphanew_sub_CV_list[idx * 2] = cluster_alphanew_sub_CV_l
        cluster_alphanew_sub_CV_list[idx * 2 + 1] = cluster_alphanew_sub_CV_r

        # Surf distance
        cluster_surf_dis[idx * 2] = cluster_surf_dis_l
        cluster_surf_dis[idx * 2 + 1] = cluster_surf_dis_r

        # centroids
        cluster_centroids[idx * 2] = cluster_centroid_l
        cluster_centroids[idx * 2 + 1] = cluster_centroid_r

    add_arrays = {}
    add_arrays['ClusterWMPG'] = cluster_WMPG_list
    add_arrays['ClusterWMPGLI'] = cluster_WMPG_LI_list
    add_arrays['ClusterNoS'] = cluster_NoS_list
    add_arrays['ClusterNoSLI'] = cluster_NoS_LI_list
    add_arrays['ClusterNoSSubCV'] = cluster_NoS_sub_CV_list
    add_arrays['ClusterFA'] = cluster_FA_list
    add_arrays['ClusterFALI'] = cluster_FA_LI_list
    add_arrays['ClusterFASubCV'] = cluster_FA_sub_CV_list
    add_arrays['ClusterUR'] = cluster_FA_list
    add_arrays['ClusterURLI'] = cluster_FA_LI_list
    add_arrays['ClusterURSubCV'] = cluster_FA_sub_CV_list
    add_arrays['ClusterAlphaN'] = cluster_alphanew_list
    add_arrays['ClusterAlphaNLI'] = cluster_alphanew_LI_list
    add_arrays['ClusterAlphaSubCV'] = cluster_alphanew_sub_CV_list
    add_arrays['ClusterSurfDis'] = cluster_surf_dis

    pd_clusters = convert_to_polydata(cluster_centroids, arrays=add_arrays)
    wma.io.write_polydata(pd_clusters, os.path.join(args.inputdir, 'clusters_controids.vtp'))

    plt.plot(cluster_surf_dis.transpose())
    plt.savefig(os.path.join(args.inputdir, 'clusters_surf_dis.png'))
    plt.close()

    with open(os.path.join(args.inputdir, 'clusters_stats.pkl'), 'wb') as f:
        pickle.dump(add_arrays, f)

if __name__ == '__main__':
    main()
