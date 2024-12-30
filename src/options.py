import argparse

parser = argparse.ArgumentParser(description="UniBrain Configuration")
parser.add_argument(
    "--model_name", type=str, default="UniBrain",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--type_model", type=str, default=None,
    help="UniBrain, etc",
)
parser.add_argument(
    "--data_path", type=str, default="../data/brain/NSD/",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--perform_group", action=argparse.BooleanOptionalAction, default=True,
    help="use grouping voxels",
)
parser.add_argument(
    "--use_fMRI_norm", action=argparse.BooleanOptionalAction, default=False,
    help="use fMRI norm or not, mean and std are calculated from training set",
)
parser.add_argument(
    "--use_fMRI_crop", action=argparse.BooleanOptionalAction, default=False,
    help="perform random crop on fMRI",
)
parser.add_argument(
    "--use_fMRI_noise", action=argparse.BooleanOptionalAction, default=False,
    help="perform random noise on fMRI",
)
parser.add_argument(
    "--noise_level",type=float,default=0.01,
    help="noise level for random noise on fMRI",
)
parser.add_argument(
    "--noise_range",type=float,default=0.05,
    help="noise range for random noise on fMRI",
)
parser.add_argument(
    "--crop_ratio",type=float,default=0.1,
    help="crop ratio for random crop on fMRI",
)
parser.add_argument(
    "--subj_list",type=int, default=[1], choices=[1,2,3,4,5,6,7,8], nargs='+',
    help="Subject index to train on",
)
parser.add_argument(
    "--subj_test_list",type=int, default=[7], choices=[1,2,3,4,5,6,7,8], nargs='+',
    help="Subject index to test on",
)
parser.add_argument(
    "--batch_size", type=int, default=50,
    help="Batch size per GPU",
)
parser.add_argument(
    "--val_batch_size", type=int, default=50,
    help="Validation batch size per GPU",
)
parser.add_argument(
    "--clip_variant",type=str,default="ViT-L/14",choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
    help='OpenAI clip variant',
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=True,
    help="Whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="UniBrain",
    help="Wandb project name",
)
parser.add_argument(
    "--resume",action=argparse.BooleanOptionalAction,default=False,
    help="Resume training from latest checkpoint, can't do it with --load_from at the same time",
)
parser.add_argument(
    "--resume_id",type=str,default=None,
    help="Run id for wandb resume",
)
parser.add_argument(
    "--load_from",type=str,default=None,
    help="load model and restart, can't do it with --resume at the same time",
)
parser.add_argument(
    "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--use_random_caption", action=argparse.BooleanOptionalAction, default=False,
    help="randomly select one caption from all five captions, instead of from three captions",
)
parser.add_argument(
    "--use_one_caption", action=argparse.BooleanOptionalAction, default=False,
    help="only select one caption from all five captions, instead of from three captions, default: 0",
)
parser.add_argument(
    "--use_mean4train", action=argparse.BooleanOptionalAction, default=False,
    help="also feed the mean of three fMRI signals into training",
)
parser.add_argument(
    "--use_scan_mix", action=argparse.BooleanOptionalAction, default=False,
    help="mix different scans of a fMRI signal",
)
parser.add_argument(
    "--mode_scan_mix", type=str, default='strength',
    help="strength or length",
)
parser.add_argument(
    "--use_global", action=argparse.BooleanOptionalAction, default=True,
    help="use global branch",
)
parser.add_argument(
    "--use_local", action=argparse.BooleanOptionalAction, default=False,
    help="use local branch",
)
parser.add_argument(
    "--use_non_linear_local", action=argparse.BooleanOptionalAction, default=False,
    help="use non linear as local model or linear model",
)
parser.add_argument(
    "--num_map_global_token", type=int, default=512, 
    help="number of global tokens",
)
parser.add_argument(
    "--in_dim", type=int, default=16384, 
    help="dimension of input features, 512 * 32",
)
parser.add_argument(
    "--latent_dim", type=int, default=32, 
    help="first hidden dim in extractor, group mapping",
)
parser.add_argument(
    "--out_dim", type=int, default=768, 
    help="dimension of output features, align with CLIP",
)
parser.add_argument(
    "--depth", type=int, default=2, 
    help="number of layers in Transformer",
)
parser.add_argument(
    "--dropout", type=float, default=0.5, 
    help="dropout ratio",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--use_fMRI_aug",action=argparse.BooleanOptionalAction,default=False,
    help="whether to use fMRI augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=600,
    help="number of epochs of training",
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
    help="Type of learning rate scheduler",
)
parser.add_argument(
    "--ckpt_interval",type=int,default=10,
    help="Save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--eval_interval",type=int,default=10,
    help="Evaluate the model every x epochs",
)
parser.add_argument(
    "--seed",type=int,default=42,
    help="Seed for reproducibility",
)
parser.add_argument(
    "--num_workers",type=int,default=0,
    help="Number of workers in dataloader"
)
parser.add_argument(
    "--max_lr",type=float,default=3e-4,
    help="Max learning rate",
)
parser.add_argument(
    "--num_keyvoxel", type=int, default=512,
    help="Number of keyvoxels in data processing",
)
parser.add_argument(
    "--num_neighbor", type=int, default=32,
    help="Number of neighbors for each keyvoxel",
)
parser.add_argument(
    "--mode_clip_loss",type=str, default="SoftClip",
    help="SoftClip or traditional Clip or Dual Clip",
)
parser.add_argument(
    "--mode_discriminator",type=str, default="non-linear",
    help="linear or non-linear",
)
parser.add_argument(
    "--mode_MutAss",type=str, default="MutAss",
    help="S_only, G_only, no_MutAss, MutAss",
)
parser.add_argument(
    "--use_detach",action=argparse.BooleanOptionalAction,default=False,
    help="whether to detach the gradient when assisting",
)
parser.add_argument(
    "--w_clip_final_image", type=float, default=1,
    help="The weight of clip loss on final image",
)
parser.add_argument(
    "--w_clip_final_text", type=float, default=1,
    help="The weight of clip loss on final text",
)
parser.add_argument(
    "--w_clip_image", type=float, default=1,
    help="The weight of clip loss on image",
)
parser.add_argument(
    "--w_clip_text", type=float, default=1,
    help="The weight of clip loss on text",
)
parser.add_argument(
    "--w_mse_final_image", type=float, default=1,
    help="The weight of mse loss on final image",
)
parser.add_argument(
    "--w_mse_final_text", type=float, default=1,
    help="The weight of mse loss on final text",
)
parser.add_argument(
    "--w_mse_image", type=float, default=1,
    help="The weight of mse loss on image",
)
parser.add_argument(
    "--w_mse_text", type=float, default=1,
    help="The weight of mse loss on text",
)
parser.add_argument(
    "--w_dis", type=float, default=1,
    help="The weight of dis loss",
)
parser.add_argument(
    "--length", type=int, default=None,
    help="Indicate dataset length",
)
parser.add_argument(
    "--autoencoder_name", type=str, default=None,
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--subj_load",type=int, default=None, choices=[1,2,3,4,5,6,7,8], nargs='+',
    help="subj want to be load in the model",
)
parser.add_argument(
    "--subj_train",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="subj to train (UMAP)",
)
parser.add_argument(
    "--subj_test",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="subj to test",
)
parser.add_argument(
    "--samples",type=int, default=None, nargs='+',
    help="Specify sample indice to reconstruction"
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--guidance_scale",type=float, default=3.5,
    help="Guidance scale for diffusion model.",
)
parser.add_argument(
    "-num_inference_steps",type=int, default=20,
    help="Number of inference steps for diffusion model.",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=16,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--plotting", action=argparse.BooleanOptionalAction, default=True,
    help="plotting all the results",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='XXX/pretrained_models/VD_weights/',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)
parser.add_argument(
    "--gpu_id", type=int, default=0,
    help="ID of the GPU to be used",
)
parser.add_argument(
    "--ckpt_from", type=str, default='last',
    help="ckpt_from ['last', 'best']",
)
parser.add_argument(
    "--text_image_ratio", type=float, default=0.5,
    help="text_image_ratio in Versatile Diffusion. Only valid when use_text=True. 0.5 means equally weight text and image, 0 means use only image",
)
parser.add_argument(
    "--test_start", type=int, default=0,
    help="test range start index",
)
parser.add_argument(
    "--test_end", type=int, default=None,
    help="test range end index, the total length of test data is 982, so max index is 981",
)
parser.add_argument(
    "--only_embeddings", action=argparse.BooleanOptionalAction, default=False,
    help="only return semantic embeddings of networks",
)
parser.add_argument(
    "--synthesis", action=argparse.BooleanOptionalAction, default=False,
    help="synthesize new fMRI signals",
)
parser.add_argument(
    "--verbose", action=argparse.BooleanOptionalAction, default=True,
    help="print more information",
)
parser.add_argument(
    "--results_path", type=str, default=None,
    help="path to reconstructed outputs",
)
parser.add_argument(
    "--use_img2img", action=argparse.BooleanOptionalAction, default=True, 
    help="use image2image or not",
)

args = parser.parse_args()