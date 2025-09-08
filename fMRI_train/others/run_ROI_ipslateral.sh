
#data=QB60-FOCL
data=$1
datastr=$data
inputfolder=./data/$datastr/

num_clusters=100
p=30
ep=50
fs=False
surf=True
gamma=0.5
runstr=${data}_ep${ep}_p${p}_n${num_clusters}_fs${fs}_surf${surf}_gamma${gamma}_ipslateral


iters=(0 1 2 3 4 5)
for i in "${iters[@]}"
do
   echo "** iteration -- $i"

   mkdir -p $inputfolder/$runstr/
   
   if [ $i == 0 ]; then
		input=$inputfolder/appendedvtp/
	else
		input=$inputfolder/$runstr/iter$((i-1))/all/
   fi

   if [ ! -f $inputfolder/$runstr/iter$i/cluster_00001.vtp ]; then
      $2 python train_ipslateral.py --pretrain True --fs True --surf True --ro True -indir $input -outdir $inputfolder/$runstr/iter$i/ -p $p --num_clusters $num_clusters --epochs_pretrain ${ep} --epochs 1
   fi
   
   out_dir=$inputfolder/$runstr/iter$i/
   # if [ $i == 0 ]; then
   #    out_dir=$inputfolder/$runstr/iter$i/sorted
   #    if [ ! -f $out_dir/cluster_00001.vtp ]; then
   #       $2 python sort_clusters_by_U_shape.py -sub_dir $inputfolder/$runstr/iter$i/ -out_dir $out_dir
   #    fi

   # else
   #    out_dir=$inputfolder/$runstr/iter$i/mapped
   #    if [ ! -f $out_dir/cluster_00001.vtp ]; then
   #       $2 python cluster_correspondence_rename.py -atl_dir $inputfolder/$runstr/iter0/sorted/ -sub_dir $inputfolder/$runstr/iter$i/ -out_dir $inputfolder/$runstr/iter$i/mapped
   #    fi
   # fi

   ROI_out_dir=$out_dir/$data
   
   if [ ! -f $ROI_out_dir/cluster_stat.csv ]; then  
      mkdir -p $ROI_out_dir
      
      if [ ! -f $ROI_out_dir/*.vtp ]; then 
         for vtk in `ls $out_dir/cluster_*.vtp`
         do
            f="$(basename $vtk)"

            # if [ ! -f $ROI_out_dir/$f ]; then
               $2 python wm_fiber_selection.py $vtk $ROI_out_dir/$f
               # exit
            # fi 
         done
      fi
   fi

   if [ ! -f $ROI_out_dir/cluster_stat.csv ]; then 
      $2 python wm_cluster_statics.py $ROI_out_dir $ROI_out_dir/cluster_stat.csv
   fi

   if [ ! -f $out_dir/cluster_stat.csv ]; then 
      $2 python wm_cluster_statics.py $out_dir $out_dir/cluster_stat.csv
   fi

   if [ ! -f $ROI_out_dir/monkey1/cluster_stat.csv ]; then
      $2 python wm_category_by_stats.py $ROI_out_dir
      $2 python wm_cluster_statics.py $ROI_out_dir/monkey0 $ROI_out_dir/monkey0/cluster_stat.csv
      $2 python wm_cluster_statics.py $ROI_out_dir/monkey1 $ROI_out_dir/monkey1/cluster_stat.csv
   fi

   if [ ! -f $out_dir/monkey-both.mrml ]; then
      $2 python wm_category_by_stats.py $out_dir
   fi


   # currentdir=$PWD
   # echo $currentdir
   # mkdir -p $inputfolder/$runstr/all_clusters
   # cd $inputfolder/$runstr/all_clusters
   # for vtk in `ls ../iter$i/*ed/cluster_*.vtp`
   # do
   #    f="$(basename $vtk)"
   #       $2 ln -s ../iter$i/*ed/$f ${f//.vtp/_iter${i}.vtp}
   # done
   # cd $currentdir

done

