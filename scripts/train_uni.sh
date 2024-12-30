export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=3600

batch_size=24
val_batch_size=24
num_epochs=600
clip_mult=1
mse_mult=100000
dis_mult=1
model_name="UniBrain"
type_model="UniBrain"
latent_dim=32

cd src/

# for multi card running
accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 --main_process_port 29526 \
main.py \
--model_name $model_name --subj_list 1 2 5 7 --subj_test_list 1 2 5 7 --wandb_log --type_model $type_model --use_global --no-use_local --use_non_linear_local \
--num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size --mode_clip_loss "SoftClip" --mode_discriminator 'non-linear' --no-use_detach \
--no-use_mean4train --no-use_scan_mix --mode_scan_mix "length" --no-use_random_caption --use_one_caption \
--num_keyvoxel 512 --num_neighbor 32 --num_map_global_token 512 --depth 2 --latent_dim $latent_dim \
--w_clip_final_image 1.0 --w_mse_final_image 100000.0 --w_clip_final_text 1.0 --w_mse_final_text 100000.0 \
--w_clip_image 1.0 --w_mse_image 100000.0 --w_clip_text 1.0 --w_mse_text 100000.0 --w_dis 1.0 \
--mode_MutAss "MutAss" \
--no-use_fMRI_crop --no-use_fMRI_noise --no-use_fMRI_norm \
--eval_interval 10 --ckpt_interval 10 \
--max_lr 1.0e-4 --num_workers 8