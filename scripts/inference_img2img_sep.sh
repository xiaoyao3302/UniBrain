img2img_strength=0.85
subj_load=7
subj_test=7
AE_name="subj07"
model_name="SepBrain"
ckpt_from="last"
gpu_id=3
type_model="SepBrain"

text_image_ratio=0.5            # 0 means use only image, 1 means use only text
guidance_scale=5.0

cd src/

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
recon.py \
--mode_MutAss "MutAss" \
--model_name $model_name --ckpt_from $ckpt_from --type_model $type_model --autoencoder_name $AE_name \
--use_img2img --img2img_strength $img2img_strength --use_global --no-use_local --use_non_linear_local \
--num_keyvoxel 512 --num_neighbor 32 --num_map_global_token 512 --depth 2 --latent_dim 32 --mode_clip_loss "SoftClip" --mode_discriminator 'non-linear' \
--no-use_fMRI_crop --no-use_fMRI_noise --no-use_fMRI_norm \
--subj_list $subj_test --subj_load $subj_load --subj_test $subj_test --length 982 \
--text_image_ratio $text_image_ratio --guidance_scale $guidance_scale \
--recons_per_sample 8 

results_path="../train_logs/"$model_name"/recon_on_subj"$subj_test"/img2img_strength_"$img2img_strength"/use_ckpt_from_"$ckpt_from"/guidance_scale_"$guidance_scale"_text_image_ratio_"$text_image_ratio

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
eval.py --results_path $results_path
