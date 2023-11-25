CUDA_VISIBLE_DEVICES=0 python test.py \
--arch emnet_enhancer \
--arch_memory emnet_memory \
--pooling_size 1 \
--mem_size 9000 \
--embed_dim 32 \
--V_feat_dim 1 \
--result_dir /mnt/data2/ckr/lowlight-image/EMNet/log_eval/ori \
--embed_dim 32 \
--weights ./pre_trained_logs/enhancer/model_lolv1.pth \
--mem_weights ./pre_trained_logs/memory/memory_lolv1.pth \
--input_dir /mnt/data2/ckr/mmdetection/data/Track1.2_testing_samples \
--testing_top_k 1 \
--save_images

#python test.py \
#--arch emnet_enhancer \
#--arch_memory emnet_memory \
#--pooling_size 1 \
#--mem_size 9000 \
#--embed_dim 32 \
#--V_feat_dim 1 \
#--result_dir /mnt/data2/ckr/lowlight-image/EMNet/log_eval/lolv2 \
#--embed_dim 32 \
#--weights ./pre_trained_logs/enhancer/model_lolv2.pth \
#--mem_weights ./pre_trained_logs/memory/memory_lolv2.pth \
#--input_dir /mnt/data2/ckr/lowlight-image/PairLIE/dataset \
#--testing_top_k 1 \
#--save_images

