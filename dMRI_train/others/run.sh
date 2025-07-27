
# data=BMFA
# data=SWM
# data=FOCL
# data=FOCL-ROI
data=$1
datastr=Monkey-QB60-$data
inputfolder=./data/$datastr/

num_clusters=1500
p=30
ep=50
runstr=${data}_2T_ep${ep}_p${p}_n${num_clusters}_bilateral

tractography=$inputfolder/tractography.vtp


##
segmentation=$inputfolder/cortex-segmentation.nii.gz
labeledtractography=$inputfolder/labeledvtp/tractography-labeled.vtp

if [ ! -f $labeledtractography ]; then 
	$2 python wm_compute_labels.py $tractography $segmentation $labeledtractography
fi

iters=(0 1 2 3 4 5)
for i in "${iters[@]}"
do
   echo "** iteration -- $i"

   mkdir -p $inputfolder/$runstr/
   
   if [ $i == 0 ]; then
		input=$inputfolder/labeledvtp/
	else
		input=$inputfolder/$runstr/iter$((i-1))/all/
      if [ ! -f $input/all_labeled.vtp ]; then
		 $2 python wm_compute_labels.py $input/all.vtp $segmentation $input/all_labeled.vtp
       $2 rm $input/all.vtp
      fi
   fi

   if [ ! -f $inputfolder/$runstr/iter$i/cluster_00001.vtp ]; then
      $2 python train.py --pretrain True --fs True --surf True --ro True -indir $input -outdir $inputfolder/$runstr/iter$i/ -p $p --num_clusters $num_clusters --epochs_pretrain ${ep} --epochs 1
   fi

   if [ $i == 0 ]; then
      out_dir=$inputfolder/$runstr/iter$i/sorted
      if [ ! -f $out_dir/cluster_00001.vtp ]; then
         $2 python sort_clusters_by_U_shape.py -sub_dir $inputfolder/$runstr/iter$i/ -out_dir $out_dir
      fi

   else
      out_dir=$inputfolder/$runstr/iter$i/mapped
      if [ ! -f $out_dir/cluster_00001.vtp ]; then
         $2 python cluster_correspondence_rename.py -atl_dir $inputfolder/$runstr/iter0/sorted/ -sub_dir $inputfolder/$runstr/iter$i/ -out_dir $inputfolder/$runstr/iter$i/mapped
      fi
   fi

   currentdir=$PWD
   echo $currentdir
   mkdir -p $inputfolder/$runstr/all_clusters
   cd $inputfolder/$runstr/all_clusters
   for vtk in `ls ../iter$i/*ed/cluster_*.vtp`
   do
      f="$(basename $vtk)"
         $2 ln -s ../iter$i/*ed/$f ${f//.vtp/_iter${i}.vtp}
   done
   cd $currentdir

done

