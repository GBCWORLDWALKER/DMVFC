#!/usr/bin/env python
import numpy
import argparse
import pandas
import os
import glob
import whitematteranalysis as wma
import vtk

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

def getcolors(number_of_files):


    step = int(100*255.0 / (number_of_files-1))
    R = numpy.array(list(range(0,100*255+1, step))) / 100.0
    G = numpy.abs(list(range(100*-127,100*128+1, step)))* 2.0 / 100.0
    B = numpy.array(list(range(100*255+1,0, -step))) / 100.0

    colors = list()
    for idx, pd in enumerate(range(number_of_files)):
        colors.append([R[idx], G[idx],B[idx]])
    colors = numpy.array(colors)

    return colors

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="",
        epilog="Written by fzhang@bwh.harvard.edu")
    
    parser.add_argument(
        'inputdir',
        help='')
    parser.add_argument(
        'inputstats',
        help='')
    parser.add_argument(
        'outputdir',
        help='')
    parser.add_argument(
        '-l', type=int, default=500,
        help='')
    parser.add_argument(
        '-u', type=float, default=1.1,
        help='')

    args = parser.parse_args()

    print("")
    print("=====input directory======\n", args.inputdir)
    print("=====input stat file======\n", args.inputstats)
    print("==========================")

    output_dir = os.path.join(args.outputdir, "l%d_u%f" % (args.l, args.u))
    os.makedirs(output_dir, exist_ok=True)

    stats = pandas.read_table(args.inputstats, delimiter=',', dtype=str)

    Len = stats['F-Len(mm)'].values.tolist()
    Len = [str(v) for v in Len]
    Len = numpy.array([float(v[:5]) for v in Len])

    Uratio = stats['Uratio'].to_numpy()
    Uratio = [str(v) for v in Uratio]
    Uratio = numpy.array([float(v[:5]) for v in Uratio])

    kept_clusters = numpy.where(numpy.logical_and(Len < args.l, Uratio < args.u))[0]
    kept_cluster_names = []
    for c_idx in kept_clusters:

        vtkf = os.path.join(args.inputdir, "cluster_%05d.vtp"%(c_idx+1))
        print(vtkf)

        outputvtkf = os.path.join(output_dir, "cluster_%05d.vtp"%(c_idx+1))
        cmd = "cp %s %s" % (vtkf, outputvtkf)
        os.system(cmd)

        kept_cluster_names.append("cluster_%05d.vtp"%(c_idx+1))

    fname = os.path.join(output_dir, 'all.mrml')
    cluster_colors = getcolors(kept_clusters.shape[0])
    wma.mrml.write(kept_cluster_names, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)


if __name__ == '__main__':
    main()
