CUDA_VISIBLE_DEVICES=0 python3 src/main.py \
    -t \
    -metrics is fid prdc \
    -data /nas/common_data/imagenet_100cls \
    -cfg ./src/configs/ImageNet/ContraGAN-256.yaml \
    -save SAVE_PATH \
    --pre_resizer "lanczos" \
    --post_resizer "friendly" \
    --eval_backbone "InceptionV3_tf" \
    -mpc \
    -hdf5 \


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/main.py -t -hdf5 -l -sync_bn -std_stat -std_max STD_MAX -std_step STD_STEP -metrics is fid prdc -ref "train" -cfg CONFIG_PATH -data DATA_PATH -save SAVE_PATH -mpc --pre_resizer "lanczos" --post_resizer "friendly" --eval_backbone "InceptionV3_tf"