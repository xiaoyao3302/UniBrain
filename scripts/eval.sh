subj_test=7
model_name="UniBrain"
ckpt_from="last"
gpu_id=3

text_image_ratio=0.5
guidance_scale=5.0

cd src/

results_path="../train_logs/"$model_name"/recon_on_subj"$subj_test"/no_img2img/use_ckpt_from_"$ckpt_from"/guidance_scale_"$guidance_scale"_text_image_ratio_"$text_image_ratio

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
eval.py --results_path $results_path