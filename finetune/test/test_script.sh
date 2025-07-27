python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle False --loss_Uratio False


python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP-Angle-Uratio_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf True --loss_angle True --loss_Uratio True --gamma 0


python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP-Angle_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf True --loss_angle True --loss_Uratio False --gamma 0


python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf True --loss_angle False --loss_Uratio False --gamma 0

export CUDA_VISIBLE_DEVICES=1; python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_Angle_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle True --loss_Uratio False --gamma 0

export CUDA_VISIBLE_DEVICES=1; python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_Uratio_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle False --loss_Uratio True --gamma 0


python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_EP_only -p 30 --num_clusters 50 --epochs_pretrain 30 --epochs 10  --loss_surf True --loss_angle False --loss_Uratio False



# subID

python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle False --loss_Uratio False --pretrain False --pretrained_net /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again/models/DGCNN_002_pretrained.pt


export CUDA_VISIBLE_DEVICES=1; python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again_ID -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle False --loss_Uratio False --loss_subID True --pretrain False --pretrained_net /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again/models/DGCNN_002_pretrained.pt


# clustering 
export CUDA_VISIBLE_DEVICES=1; python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP_gamma100 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf True --loss_angle False --loss_Uratio False --gamma 100 --pretrain False --pretrained_net /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP_gamma0/models/DGCNN_001_pretrained.pt 




python cluster_correspondence_rename.py -atl_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev -sub_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP-Angle-Uratio_gamma0/DFC-DGCNN_001-POG_EP-Angle-Uratio_gamma0/ -out_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP-Angle-Uratio_gamma0/DFC-DGCNN_001-POG_EP-Angle-Uratio_gamma0_in_org

python cluster_correspondence_rename.py -atl_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev -sub_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP_gamma0/DFC-DGCNN_001-POG_EP_gamma0/ -out_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_EP_gamma0/DFC-DGCNN_001-POG_EP_gamma0_in_org

python cluster_correspondence_rename.py -atl_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev -sub_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_Angle_gamma0/DFC-DGCNN_001-POG_Angle_gamma0/ -out_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_Angle_gamma0/DFC-DGCNN_001-POG_Angle_gamma0_in_org

python cluster_correspondence_rename.py -atl_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev -sub_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_Uratio_gamma0/DFC-DGCNN_001-POG_Uratio_gamma0/ -out_dir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_Uratio_gamma0/DFC-DGCNN_001-POG_Uratio_gamma0_in_org




# both

python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_both_DFC_prev -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle False --loss_Uratio False --gamma 0

python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_both_DFC_EP_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf True --loss_angle False --loss_Uratio False --gamma 0

export CUDA_VISIBLE_DEVICES=1; python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_both_DFC_Angle_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf False --loss_angle True --loss_Uratio False --gamma 0

export CUDA_VISIBLE_DEVICES=1; python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_boyh_EP-Angle-Uratio_gamma0 -p 30 --num_clusters 50 --epochs_pretrain 300 --epochs 10  --loss_surf True --loss_angle True --loss_Uratio True --gamma 0





python cluster_correspondence_rename.py -atl_dir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_both_DFC_prev/DFC-DGCNN_001-POG_both_DFC_prev/ -sub_dir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_both_DFC_EP_gamma0/DFC-DGCNN_001-POG_both_DFC_EP_gamma0/ -out_dir /mnt/data0/fan/SWM-DFC/data/POG_both_n50/POG_both_DFC_EP_gamma0/DFC-DGCNN_001-POG_both_DFC_EP_gamma0_in_org





python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/test_cluster6/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/test_cluster6/appendedvtp/test -p 30 --num_clusters 50 --epochs_pretrain 2 --epochs 1  --loss_surf False --loss_angle False --loss_Uratio False --loss_subID True





python train_bilateral_Fan.py -indir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/appendedvtp/ -outdir /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again_debug_empty_clusters -p 30 --num_clusters 300 --epochs_pretrain 10 --epochs 10  --loss_surf False --loss_angle False --loss_Uratio False 

# --pretrain False --pretrained_net /mnt/data0/fan/SWM-DFC/data/POG_l_n50/POG_DFC_prev_again/models/DGCNN_002_pretrained.pt




