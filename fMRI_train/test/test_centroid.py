
from utils import fibers
from utils.io import convert_fiber_to_array
from utils import clusters
import numpy
import vtk
import whitematteranalysis as wma

def convert_to_polydata(fiber_array):
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

    return outpd


vtk = 'cluster_00003.vtp'

input_pd, x_array, d_roi ,fiber_surf_dk, feat_Uratio, feat_v2, feat_v3, subID = \
            convert_fiber_to_array(vtk, numberOfFiberPoints=30, preproces=False)


print(x_array[:, 0, 0])
predicted = numpy.zeros(x_array.shape[0])
print(predicted)
cluster_centroids, cluster_reordered_fibers = cluster.cluster_centroids(1, predicted, x_array)

print(cluster_centroids)

pd = convert_to_polydata(cluster_centroids)
wma.io.write_polydata(pd, 'cluster_00003_centoid.vtp')

wma.io.write_polydata(input_pd, 'cluster_00003_order.vtp')



vtk_array = vtk.vtkDoubleArray()
vtk_array.SetName('Point_Seq_reorder')

inpointdata = input_pd.GetPointData()
point_data_array_indices = list(range(inpointdata.GetNumberOfArrays()))
roi_list = []
for idx in point_data_array_indices:
    array = inpointdata.GetArray(idx)
    if array.GetName() == 'Point_Seq':
        input_pd.GetLines().InitTraversal()
        for lidx in range(0, input_pd.GetNumberOfLines()):
            ptids = vtk.vtkIdList()
            input_pd.GetLines().GetNextCell(ptids)

            roi_line = -numpy.ones(ptids.GetNumberOfIds())  # roi_line = -numpy.ones(ptids.GetNumberOfIds())
            for ind, pidx in enumerate(range(ptids.GetNumberOfIds())):  # TODO now, we only consider ep.
                roi_line[ind] = array.GetTuple(ptids.GetId(pidx))[0]

            if cluster_reordered_fibers[0][lidx] == 1:
                roi_line = numpy.flip(roi_line)

            for ind, pidx in enumerate(range(ptids.GetNumberOfIds())):  # TODO now, we only consider ep.
                vtk_array.InsertNextTuple1(roi_line[ind])

inpointdata.AddArray(vtk_array)
inpointdata.Update()

wma.io.write_polydata(input_pd, 'cluster_00003_reorder.vtp')