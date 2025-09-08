# python wm_compute_labels.py /home/annabelchen/PycharmProjects/SWMA/files/UKF-1T-GM-0p08-0p05-0p01-l100-r1-regionfiltered.vtp /home/annabelchen/PycharmProjects/SWMA/files/nouchine_convert_b0space.nii.gz /home/annabelchen/PycharmProjects/SWMA/files/processed_region.vtk
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

try:
    from joblib import Parallel, delayed
except:
    print("<wm_laterality.py> Error importing joblib package\n")
    raise


def fiber_selection(inpd, pointdataname):
   
    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()
    
    mask = numpy.zeros(inpd.GetNumberOfLines())

    ROI_arrary = inpointdata.GetArray(pointdataname)

    inpd.GetLines().InitTraversal()
    # print(inpd.GetNumberOfLines())
    for lidx in range(0, inpd.GetNumberOfLines()):

        ptids = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(ptids)
        
        line_region = []
        for pidx in range(0, ptids.GetNumberOfIds()):
            point = inpoints.GetPoint(ptids.GetId(pidx))
            line_region.append(ROI_arrary.GetTuple(ptids.GetId(pidx))[0])
        
            uni_region = numpy.unique(line_region)
            if len(uni_region) == 2:
                mask[lidx] = 1

    return mask

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Applies preprocessing to input directory. Downsamples, removes short fibers. Preserves tensors and scalar point data along retained fibers.",
        epilog="Written by Fan Zhang, fzhang@bwh.harvard.edu")
    
    parser.add_argument(
        'inputdir',
        help='Contains whole-brain tractography as vtkPolyData file(s).')

    parser.add_argument(
        'outputdir',
        help='The output directory should be a new empty directory. It will be created if needed.')


    args = parser.parse_args()

    print("")
    print("=====input directory======\n", args.inputdir)
    print("=====output directory=====\n", args.outputdir)
    print("==========================")

    # =======================================================================
    # Above this line is argument parsing. Below this line is the pipeline.
    # =======================================================================

    os.makedirs(args.outputdir, exist_ok=True)
    input_mask = "{0}/cluster_*.vtp".format(os.path.join(args.inputdir))
    subject_clusters = sorted(glob.glob(input_mask))

    for cluster in subject_clusters:
        filename = os.path.basename(cluster)
        inpd = wma.io.read_polydata(cluster)
        # print(inpd)
        mask = fiber_selection(inpd, "seletion_ROI_label")

        print('%s, totalNoS: %s' % (filename, len(mask)))
        print('   Saving kept: NoS %s' % numpy.sum(mask == 1))
        pd_ds = wma.filter.mask(inpd, mask==1, preserve_point_data=True, preserve_cell_data=True, verbose=False)
        wma.io.write_polydata(pd_ds, os.path.join(args.outputdir, filename))

if __name__ == '__main__':
    main()
