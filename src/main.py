import os
import sys
import torch
from accelerate import Accelerator, DeepSpeedPlugin

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# Custom models and functions 
from models import Clipper, SepBrain, UniBrain
from nsd_access import NSDAccess
from trainer import *
from options import args
import utils
import pdb


def config_multi_gpu():
    # Multi-GPU config
    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_clipping=1.0)
    accelerator = Accelerator(split_batches=False, mixed_precision='no', deepspeed_plugin=deepspeed_plugin)  
    accelerator.print("PID of this process =",os.getpid())
    device = accelerator.device
    accelerator.print("device:",device)
    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    accelerator.print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    accelerator.print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size)

    return accelerator, device, local_rank

def prepare_CLIP(args, device):
    # Prepare CLIP
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_size = clip_sizes[args.clip_variant]                                           # default: ViT-L/14

    print("Using hidden layer CLIP space (Versatile Diffusion)")
    if not args.norm_embs:
        print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
    clip_extractor = Clipper(args.clip_variant, device=device, hidden_state=True, norm_embs=args.norm_embs)

    out_dim_image = 257 * clip_size # 257*768 = 197376
    out_dim_text  = 77  * clip_size # 77*768  = 59136

    print("clip_extractor loaded.")
    print("out_dim_image:",out_dim_image)
    print("out_dim_text:", out_dim_text)

    return clip_extractor, out_dim_image, out_dim_text

def prepare_voxel2clip(args, device):
    # Prepare voxel2clip
    if args.type_model == 'SepBrain':
        voxel2clip = SepBrain(args).to(device)
    elif args.type_model == 'UniBrain':
        voxel2clip = UniBrain(args).to(device)
    else:
        assert args.type_model == 'UniBrain', 'unknown model'
    
    # if args.adapting: # reset-tuning
    #     # Only let the parameters of embedder and builder in the voxel2clip trainable, keeping other parameters frozen
    #     voxel2clip.requires_grad_(False)
    #     voxel2clip.embedder[str(args.subj_target)].requires_grad_(True)
    #     voxel2clip.builder[str(args.subj_target)].requires_grad_(True)

    print("voxel2clip loaded.")
    print("params of voxel2clip:")
    utils.count_params(voxel2clip)
    
    return voxel2clip

def prepare_coco(args):
    # Preload coco captions
    nsda = NSDAccess(args.data_path)
    coco_73k = list(range(0, 73000))
    prompts_list = nsda.read_image_coco_info(coco_73k,info_type='captions')
    print("coco captions loaded.")

    return prompts_list

def prepare_trainer(args, accelerator, voxel2clip, clip_extractor, prompts_list, device):
    # 1. fully supervised: train on all of four subjects, test on all of four subjects
    # 2. DG: train on three subjects, test on the rest subject
    trainer = Trainer_UniBrain(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)
    
    return trainer


def main():
    accelerator, device, local_rank = config_multi_gpu()
    if local_rank != 0: # suppress print for non-local_rank=0
        sys.stdout = open(os.devnull, 'w')

    # need non-deterministic CuDNN for conv3D to work
    utils.seed_everything(args.seed, cudnn_deterministic=False)

    # learning rate will be changed by "acclerate" based on number of processes(GPUs)
    args.max_lr *= accelerator.num_processes

    # Prepare CLIP
    clip_extractor, out_dim_image, out_dim_text = prepare_CLIP(args, device)

    # Prepare voxel2clip: fMRI encoder, including the embedder, the builder and the Translator
    voxel2clip = prepare_voxel2clip(args, device)

    # Prepare coco captions
    prompts_list = prepare_coco(args)

    # Init Trainer
    trainer = prepare_trainer(args, accelerator, voxel2clip, clip_extractor, prompts_list, device)
    trainer.prepare_wandb(local_rank, args)
    # trainer.prepare_multi_gpu()

    if local_rank == 0:
        outdir = os.path.abspath(f'../train_logs/{args.model_name}')
        output_file = outdir + '/args.txt'
        with open(output_file, 'w') as f:
            for key, value in vars(args).items():
                f.write(f'{key}: {value}\n')

    # Train or Adapt
    trainer.train(local_rank)

    print("\n===Finished!===\n")

if __name__ == '__main__':
    main()