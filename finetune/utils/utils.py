import numpy
import glob
import vtk
import argparse


def print_both(f, text):
    print(text)
    f.write(text + '\n')


def list_files(input_dir,str):
    # Find input files
    input_mask = ("{0}/"+str+"*").format(input_dir)
    input_pd_fnames = glob.glob(input_mask)
    input_pd_fnames = sorted(input_pd_fnames)
    return(input_pd_fnames)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    inpd.GetPointData().update()

    return inpd


def surf_encoding(fiber_surf_dk, surf_map):

    fiber_surfs = fiber_surf_dk.astype(int)
    surf_labels = numpy.unique(fiber_surfs)
    for surf_label in surf_labels:
        fiber_surfs[numpy.where(fiber_surfs == surf_label)] = surf_map[1][surf_map[0] == surf_label]

    ds_surf_onehot_dk = numpy.zeros((len(fiber_surfs), len(numpy.unique(surf_map[1])))).astype(numpy.float32)
    for s in range(len(fiber_surfs)):
        ds_surf_onehot_dk[s, fiber_surfs[s]] = 1

    return ds_surf_onehot_dk


def params_to_str(d):
    params_str = ""
    for key, value in d.items():
        if key == "log" or key == "writer":
            continue
        if key == "model_files":
            value = value[0] + "; " + value[1] + "; " + value[2]

        params_str += "{:<30}".format(str(key)) + ': \t' + str(value) + "\n"

    return params_str


def add_vtk_arry(inpd, array_name, array_value):
    if numpy.isscalar(array_value):
        # For scalar data
        vtk_array = vtk.vtkDoubleArray()
        vtk_array.SetName(array_name)
        vtk_array.InsertNextValue(array_value)
    elif isinstance(array_value, (list, tuple, numpy.ndarray)):
        if len(array_value) == 0:
            print(f"Warning: Empty array for {array_name}")
            return inpd
        if isinstance(array_value[0], (list, tuple, numpy.ndarray)):
            # For multi-dimensional data (like colors)
            vtk_array = vtk.vtkFloatArray()
            vtk_array.SetName(array_name)
            vtk_array.SetNumberOfComponents(len(array_value[0]))
            for val in array_value:
                vtk_array.InsertNextTuple(val)
        else:
            # For 1D array data
            vtk_array = vtk.vtkDoubleArray()
            vtk_array.SetName(array_name)
            for val in array_value:
                vtk_array.InsertNextValue(val)
    else:
        print(f"Error: Unsupported data type for {array_name}")
        return inpd

    if inpd.GetPointData().GetNumberOfTuples() == vtk_array.GetNumberOfTuples() or vtk_array.GetNumberOfTuples() == 1:
        inpd.GetPointData().AddArray(vtk_array)
    elif inpd.GetCellData().GetNumberOfTuples() == vtk_array.GetNumberOfTuples() or vtk_array.GetNumberOfTuples() == 1:
        inpd.GetCellData().AddArray(vtk_array)
    else:
        print(f"Error: Array size does not match point or cell count for {array_name}")
    
    return inpd


def convert_to_polydata(fiber_array, arrays=None):
    """Convert fiber array to vtkPolyData object."""

    outpd = vtk.vtkPolyData()
    outpoints = vtk.vtkPoints()
    outlines = vtk.vtkCellArray()

    fiber_array_ = numpy.array(fiber_array)
    fiber_array_[:, :, 0] = -fiber_array_[:, :, 0]
    fiber_array = numpy.concatenate((fiber_array, fiber_array_), axis=0)
    for key, value in arrays.items():
        arrays[key] = value + value

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

    for key, value in arrays.items():
        value = numpy.array(value)
        outpd = add_vtk_arry(outpd, key, value)

    return outpd

def fiber_reorder(inpd, reorder, verbose=False):

    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()

    # output and temporary objects
    ptids = vtk.vtkIdList()
    outpd = vtk.vtkPolyData()
    outlines = vtk.vtkCellArray()
    outpoints = vtk.vtkPoints()
    outpointdata = outpd.GetPointData()

    tensor_names = []

    # check for point data arrays to keep
    preserve_point_data = True
    if inpointdata.GetNumberOfArrays() > 0:
        point_data_array_indices = list(range(inpointdata.GetNumberOfArrays()))
        for idx in point_data_array_indices:
            array = inpointdata.GetArray(idx)
            out_array = vtk.vtkFloatArray()
            out_array.SetNumberOfComponents(array.GetNumberOfComponents())
            out_array.SetName(array.GetName())
            if verbose:
                print("Point data array found:", array.GetName(), array.GetNumberOfComponents())
            outpointdata.AddArray(out_array)
            # make sure some scalars are active so rendering works
            # outpd.GetPointData().SetActiveScalars(array.GetName())
            # keep track of tensors to choose which is active
            if array.GetNumberOfComponents() == 9:
                tensor_names.append(array.GetName())
    else:
        preserve_point_data = False

    # For Slicer: First set one of the expected tensor arrays as default for vis
    tensors_labeled = False
    for name in tensor_names:
        if name == "tensors":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("tensors"))
            tensors_labeled = True
        if name == "Tensors":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("Tensors"))
            tensors_labeled = True
        if name == "tensor1":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("tensor1"))
            tensors_labeled = True
        if name == "Tensor1":
            outpd.GetPointData().SetTensors(outpd.GetPointData().GetArray("Tensor1"))
            tensors_labeled = True
    if not tensors_labeled:
        if len(tensor_names) > 0:
            print("Data has unexpected tensor name(s). Unable to set active for visualization:", tensor_names)

    # now set cell data visualization inactive.
    outpd.GetCellData().SetActiveScalars(None)

    # loop over lines
    inpd.GetLines().InitTraversal()
    outlines.InitTraversal()
    for lidx in range(0, inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)

        if verbose:
            if lidx % 100 == 0:
                print("<filter.py> Line:", lidx, "/", inpd.GetNumberOfLines())

        # get points for each ptid and add to output polydata
        cellptids = vtk.vtkIdList()

        if reorder[lidx] == 0:
            ptid_list = range(0, ptids.GetNumberOfIds())
        else:
            ptid_list = range(ptids.GetNumberOfIds()-1, -1, -1)

        for out_pidx, in_pidx in enumerate(ptid_list):
            point = inpoints.GetPoint(ptids.GetId(in_pidx))
            idx = outpoints.InsertNextPoint(point)
            cellptids.InsertNextId(idx)
            if preserve_point_data:
                for idx in point_data_array_indices:
                    array = inpointdata.GetArray(idx)
                    out_array = outpointdata.GetArray(idx)
                    if array.GetName() == "FiberPointSeq" and reorder[lidx] == 0:
                        out_array.InsertNextTuple1(ptids.GetNumberOfIds()-array.GetTuple(ptids.GetId(in_pidx))[0]-1)
                    else:
                        out_array.InsertNextTuple(array.GetTuple(ptids.GetId(in_pidx)))

        outlines.InsertNextCell(cellptids)

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)

    return outpd