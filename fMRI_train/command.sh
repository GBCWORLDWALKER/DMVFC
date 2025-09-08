
nohup python -u src/DFC/SWM_DFC-main/test_bilateral_Fan.py \
    -inputDirectory data/full_brain/test_dfc \
    -outdir data/full_brain/test_dfc_out/funct \
    -trained_net /data06/jinwang/isbi/data/full_brain/train_dfc_out/models/DGCNN_050_final_k800.pt\
    -surf False \
    -batch_size 1024 \
    -p 25 \
    > test_bilateral_Fan.log 2>&1 &
nohup python -u src/DFC/SWM_DFC-main/test_bilateral_Fan.py \
    -inputDirectory data/full_brain/test_dfc \
    -outdir data/full_brain/test_dfc_out/norm \
    -trained_net /data06/jinwang/isbi/data/full_brain/train_dfc_out/models/DGCNN_049_final_k800.pt\
    -surf False \
    -batch_size 1024 \
    -p 25 \
    -gpu 1\
    > test_bilateral_Fan_norm.log 2>&1 &