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
        description="Applies preprocessing to input directory. Downsamples, removes short fibers. Preserves tensors and scalar point data along retained fibers.",
        epilog="Written by Lauren O\'Donnell, odonnell@bwh.harvard.edu")
    
    parser.add_argument(
        'inputdir',
        help='')

    args = parser.parse_args()
    
    print("Input File", args.inputdir)
    
    print("")
    print("=====input directory======\n", args.inputdir)
    print("==========================")

    
    # =======================================================================
    # Above this line is argument parsing. Below this line is the pipeline.
    # =======================================================================
    # def list_files(input_dir,str):
	   #  # Find input files
	   #  input_mask = ("{0}/"+str+"*.vtp").format(input_dir)
	   #  input_pd_fnames = glob.glob(input_mask)
	   #  input_pd_fnames = sorted(input_pd_fnames)
	   #  return(input_pd_fnames)


    # vtkfiles = list_files(args.inputdir, 'cluster')


    stasfile = os.path.join(args.inputdir, 'cluster_stat.csv')

    stats = pandas.read_table(stasfile, delimiter=',')

    NoS = stats['NoS'].to_numpy()
    clusters = stats['cluster'].to_numpy()

    kept_NoS = numpy.where(NoS > 20)[0]

    Monkey1 = stats['Monkey1 (fiber percent)'].to_numpy()

    kept_Monkey1 = numpy.where((NoS > 20) & (Monkey1 >= 0.9))[0]
    kept_Monkey1_clusters = clusters[kept_Monkey1]

    kept_Monkey2 = numpy.where((NoS > 20) & (Monkey1 <= 0.1))[0]
    kept_Monkey2_clusters = clusters[kept_Monkey2]

    kept_both = numpy.where((NoS > 20) & (Monkey1 > 0.1) & (Monkey1 < 0.9))[0]
    kept_both_clusters = clusters[kept_both]

    fname = os.path.join(args.inputdir, 'monkey-1.mrml')
    cluster_colors = getcolors(len(kept_Monkey1_clusters))
    wma.mrml.write(kept_Monkey1_clusters, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)

    fname = os.path.join(args.inputdir, 'monkey-2.mrml')
    cluster_colors = getcolors(len(kept_Monkey2_clusters))
    wma.mrml.write(kept_Monkey2_clusters, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)

    fname = os.path.join(args.inputdir, 'monkey-both.mrml')
    cluster_colors = getcolors(len(kept_both_clusters))
    wma.mrml.write(kept_both_clusters, numpy.around(numpy.array(cluster_colors), decimals=3), fname, ratio=1)

    print(kept_both_clusters)
    
    for filename in kept_both_clusters:

        vtkf = os.path.join(args.inputdir, filename)
        print(filename)

        inpd = wma.io.read_polydata(vtkf)

        mask = line_regions(inpd)

        print('Total NoS: %s' % len(mask))

        os.makedirs(os.path.join(args.inputdir, 'monkey0'), exist_ok=True)
        os.makedirs(os.path.join(args.inputdir, 'monkey1'), exist_ok=True)

        print('Saving monkey 0 : NoS %s' % numpy.sum(mask==0))
        pd_ds = wma.filter.mask(inpd, mask==0, preserve_point_data=True, preserve_cell_data=True, verbose=False)
        wma.io.write_polydata(pd_ds, os.path.join(args.inputdir, 'monkey0', filename))

        print('Saving monkey 1 : NoS %s' % numpy.sum(mask==1))
        pd_ds = wma.filter.mask(inpd, mask==1, preserve_point_data=True, preserve_cell_data=True, verbose=False)
        wma.io.write_polydata(pd_ds, os.path.join(args.inputdir, 'monkey1', filename))
        
        print("save to", os.path.join(args.inputdir, 'monkey1', filename))


if __name__ == '__main__':
    main()
