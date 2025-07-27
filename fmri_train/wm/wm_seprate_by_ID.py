def line_regions(inpd):

    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()

    mask = numpy.zeros(inpd.GetNumberOfLines())

    ROI_arrary = inpointdata.GetArray("cluster_idx")

    inpd.GetLines().InitTraversal()
    for lidx in range(0, inpd.GetNumberOfLines()):

        ptids = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(ptids)

        mask[lidx] = ROI_arrary.GetTuple(ptids.GetId(0))[0]

    return mask

import argparse
import os
import vtk
import numpy
import glob

try:
    import whitematteranalysis as wma
except:
    print("Error importing white matter analysis package\n")
    raise

#-----------------
# Parse arguments
#-----------------
parser = argparse.ArgumentParser(
    description="Convert a fiber tract or cluster (vtk) to a voxel-wise fiber density image (nii.gz). ",
    epilog="Written by Fan Zhang")

parser.add_argument("-v", "--version",
    action="version", default=argparse.SUPPRESS,
    version='1.0',
    help="Show program's version number and exit")

parser.add_argument(
    'inputfolder',
    help='Input folder.')
parser.add_argument(
    'outputfolder',
    help='Output folder.')

args = parser.parse_args()

def list_files(input_dir,str):
    # Find input files
    input_mask = ("{0}/"+str+"*.vtp").format(input_dir)
    input_pd_fnames = glob.glob(input_mask)
    input_pd_fnames = sorted(input_pd_fnames)
    return(input_pd_fnames)

vtkfiles = list_files(args.inputfolder, 'cluster')

for vtkf in vtkfiles:

    filename = os.path.basename(vtkf)
    print(filename)

    inpd = wma.io.read_polydata(vtkf)

    mask = line_regions(inpd)

    print('Total NoS: %s' % len(mask))

    os.makedirs(os.path.join(args.outputfolder, 'monkey0'), exist_ok=True)
    os.makedirs(os.path.join(args.outputfolder, 'monkey1'), exist_ok=True)

    print('Saving monkey 0 : NoS %s' % numpy.sum(mask==0))
    pd_ds = wma.filter.mask(inpd, mask==0, preserve_point_data=True, preserve_cell_data=True, verbose=False)
    wma.io.write_polydata(pd_ds, os.path.join(args.outputfolder, 'monkey0', filename))

    print('Saving monkey 0 : NoS %s' % numpy.sum(mask==1))
    pd_ds = wma.filter.mask(inpd, mask==1, preserve_point_data=True, preserve_cell_data=True, verbose=False)
    wma.io.write_polydata(pd_ds, os.path.join(args.outputfolder, 'monkey1', filename))






