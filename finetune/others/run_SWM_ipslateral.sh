
#data=QB60-FOCL
data=$1
datastr=$data
inputfolder=./data/$datastr/

num_clusters=3000
p=30
ep=50
runstr=${data}_ep${ep}_p${p}_n${num_clusters}_ipslateral


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


   if [ ! -f $out_dir/cluster_stat.csv ]; then 
      $2 python wm_cluster_statics.py $out_dir $out_dir/cluster_stat.csv
   fi

done

