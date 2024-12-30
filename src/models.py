import clip
import torchsnooper
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial
import numpy as np
from torchvision import transforms
from einops import repeat
from perceiver import PerceiverResampler
from augmentation import fMRI_random_crop, fMRI_random_noise
import pdb


def add_hooks(module, parent_name=''):
    module_name = module.__class__.__name__
    if parent_name:
        module_name = f'{parent_name}.{module_name}'

    module.register_forward_hook(lambda mod, inp, out: forward_hook(mod, inp, out, module_name))
    module.register_backward_hook(lambda mod, inp, out: backward_hook(mod, inp, out, module_name))

    for name, child in module.named_children():
        add_hooks(child, parent_name=f'{module_name}.{name}')

def forward_hook(module, input, output, name):
    if output.isnan().any():
        print(f"NaN detected in forward pass in module: {name}")
        print(f"Input: {input}")
        print(f"Output: {output}")

def backward_hook(module, grad_input, grad_output, name):
    if any(tensor is not None and torch.isnan(tensor).any() for tensor in [*grad_input, *grad_output]):
        print(f"NaN detected in backward pass in module: {name}")
        print(f"Grad Input: {grad_input}")
        print(f"Grad Output: {grad_output}")

class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            image_encoder = image_encoder.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.image_encoder = image_encoder

            text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            text_encoder = text_encoder.to(device)
            for param in text_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.text_encoder = text_encoder
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size, antialias=None)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        # clamp_embs is set as False by default
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:  # norm_embs is set as True by default, we need to align the normed feature from CLIP and from fMRI encoder
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb
    
    def embed_text(self, prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return embeds

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
            )
        prompt_embeds = normalize_embeddings(prompt_embeds)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = prompt_embeds.shape
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)


# ======================================================================================================= #
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    

class CrossAttention(nn.Module):
    def __init__(self, patch_embed_dim=768, hidden_size=768, num_latents=256+77, depth=2):
        super().__init__()
    
        self.ln = nn.LayerNorm(patch_embed_dim)
        self.proj = nn.Linear(
            patch_embed_dim, hidden_size
        )

        self.perceiver = PerceiverResampler(
            dim = patch_embed_dim,
            dim_head = 96,
            depth = depth,
            heads = 16,
            num_latents = num_latents,
            num_media_embeds = 1
        )

    def forward(self, image_features):
        image_features = self.ln(image_features)
        image_features = self.perceiver(image_features)
        return self.proj(image_features)


class SepBrain(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
            the resolution of fMRI is low, fMRI is hard to reconstruct image
            the CLIP text feature is high-level, lacks fine-grained feature
            we first align fMRI with text, and align fMRI with image
            then, we feed the pred image token and text token to a shared Transformer to enable the two modalities to learn from each other
        """
        self.num_keyvoxel = args.num_keyvoxel
        self.num_neighbor = args.num_neighbor
        self.subj_list = args.subj_list
        self.num_map_global_token = args.num_map_global_token                   # map the global feature to several tokens
        self.in_dim = args.in_dim                                               # flattern the fMRI signal, 512 * 32 = 16382
        self.out_dim = args.out_dim                                             # map to CLIP, 768
        self.latent_dim = args.latent_dim
        self.dropout = args.dropout
        self.depth = args.depth                                                 # depth of Transformer

        # =================================================================feature extractor================================================================= #

        # -------------------------------------------------------------------------------------------------------------------------------------- #
        # text branch
        # -------------------------------------------------------------------------------------------------------------------------------------- #

        # cat the feature from different groups, then perform mapping to global feature
        self.text_group_global_mapping = nn.ModuleDict({
            str(subj): nn.Linear(self.num_neighbor, self.latent_dim) for subj in self.subj_list
        })

        self.text_global_mapping = nn.ModuleDict({
            str(subj): nn.Linear(self.num_keyvoxel * self.latent_dim, self.out_dim) for subj in self.subj_list
        })
        
        # map the global feature to tokens
        self.text_global_token_mapping = nn.ModuleDict({
            str(subj): nn.Sequential(
                nn.Linear(1, self.num_map_global_token),
                nn.LayerNorm(self.num_map_global_token),
                nn.Dropout(0.5),
            ) for subj in self.subj_list
        })

        # -------------------------------------------------------------------------------------------------------------------------------------- #
        # image branch
        # -------------------------------------------------------------------------------------------------------------------------------------- #

        # cat the feature from different groups, then perform mapping to global feature
        self.image_group_global_mapping = nn.ModuleDict({
            str(subj): nn.Linear(self.num_neighbor, self.latent_dim) for subj in self.subj_list
        })

        self.image_global_mapping = nn.ModuleDict({
            str(subj): nn.Linear(self.num_keyvoxel * self.latent_dim, self.out_dim) for subj in self.subj_list
        })
        
        # map the global feature to tokens
        self.image_global_token_mapping = nn.ModuleDict({
            str(subj): nn.Sequential(
                nn.Linear(1, self.num_map_global_token),
                nn.LayerNorm(self.num_map_global_token),
                nn.Dropout(0.5),
            ) for subj in self.subj_list
        })

        # -------------------------------------------------------------------------------------------------------------------------------------- #
        # unified branch
        # -------------------------------------------------------------------------------------------------------------------------------------- #

        # cat the feature from different groups, then perform mapping to global feature
        self.unified_group_global_mapping = nn.ModuleDict({
            str(subj): nn.Linear(self.num_neighbor, self.latent_dim) for subj in self.subj_list
        })

        self.unified_global_mapping = nn.ModuleDict({
            str(subj): nn.Linear(self.num_keyvoxel * self.latent_dim, self.out_dim) for subj in self.subj_list
        })
        
        # map the global feature to tokens
        self.unified_global_token_mapping = nn.ModuleDict({
            str(subj): nn.Sequential(
                nn.Linear(1, self.num_map_global_token),
                nn.LayerNorm(self.num_map_global_token),
                nn.Dropout(0.5),
            ) for subj in self.subj_list
        })

        # =================================================================unified encoder================================================================= #
        # text encoder
        self.TextEncoder = CrossAttention(patch_embed_dim=self.out_dim, hidden_size=self.out_dim, num_latents=77, depth=self.depth)     
        
        # image encoder
        self.ImageEncoder = CrossAttention(patch_embed_dim=self.out_dim, hidden_size=self.out_dim, num_latents=257, depth=self.depth)     

        # unified encoder
        self.UniEncoder = CrossAttention(patch_embed_dim=self.out_dim, hidden_size=self.out_dim, num_latents=257+77, depth=self.depth)    

    def forward(self, x, alpha=0.0):
        out_list = []

        if len(x) == 2 and type(x) is tuple:
            subj_list = x[1].tolist()
            x = x[0]
        else:
            subj_list = self.subj_list

        batch_size, num_keyvoxel, num_neighbor = x.shape
        
        # divide into several subjects
        x = torch.chunk(x, len(subj_list))                                                                                                      # 4 x [B * num_keyvoxel * num_neighbor]
        
        # save feature
        text_global_token_feature_list = []
        image_global_token_feature_list = []
        unified_global_token_feature_list = []
        
        for ii, subj_i in enumerate(subj_list):
            current_x = x[ii]
            current_batch_size = current_x.shape[0]
            # ============================================================= #
            # extractor
            # ============================================================= #

            # ------------------------------------------------------------- #
            # text feature
            
            # extract feature for each group
            current_x_global_text = self.text_group_global_mapping[str(subj_i)](current_x.reshape(-1, num_neighbor)).reshape(current_batch_size, num_keyvoxel, -1)      # B * num_keyvoxel * num_neighbor --> B * num_keyvoxel * 8
            
            # map to global feature
            global_text_feature = current_x_global_text.reshape(current_batch_size, -1)                                                    # B * num_keyvoxel * 8 --> B * (num_keyvoxel * 8)
            global_text_feature = self.text_global_mapping[str(subj_i)](global_text_feature)                                               # B * (num_keyvoxel * 8) --> B * 768

            # map to tokens
            text_global_token_feature = global_text_feature.unsqueeze(-1)                                                                  # B * 768 --> B * 768 * 1
            text_global_token_feature = self.text_global_token_mapping[str(subj_i)](text_global_token_feature)                             # B * 768 * 1 --> B * 768 * num_token
            text_global_token_feature = text_global_token_feature.transpose(2, 1)                                                          # B * 768 * num_token --> B * num_token * 768
            
            text_global_token_feature_list.append(text_global_token_feature)

            # ------------------------------------------------------------- #
            # image feature

            # extract feature for each group
            current_x_global_image = self.image_group_global_mapping[str(subj_i)](current_x.reshape(-1, num_neighbor)).reshape(current_batch_size, num_keyvoxel, -1)      # B * num_keyvoxel * num_neighbor --> B * num_keyvoxel * 8

            # map to global feature
            global_image_feature = current_x_global_image.reshape(current_batch_size, -1)                                                  # B * num_keyvoxel * 8 --> B * (num_keyvoxel * 8)
            global_image_feature = self.image_global_mapping[str(subj_i)](global_image_feature)                                            # B * (num_keyvoxel * 8) --> B * 768

            # map to tokens
            image_global_token_feature = global_image_feature.unsqueeze(-1)                                                                # B * 768 --> B * 768 * 1
            image_global_token_feature = self.image_global_token_mapping[str(subj_i)](image_global_token_feature)                          # B * 768 * 1 --> B * 768 * num_token
            image_global_token_feature = image_global_token_feature.transpose(2, 1)                                                        # B * 768 * num_token --> B * num_token * 768
            
            image_global_token_feature_list.append(image_global_token_feature)

            # ------------------------------------------------------------- #
            # unified feature

            # extract feature for each group
            current_x_global_unified = self.unified_group_global_mapping[str(subj_i)](current_x.reshape(-1, num_neighbor)).reshape(current_batch_size, num_keyvoxel, -1)      # B * num_keyvoxel * num_neighbor --> B * num_keyvoxel * 8
            
            # map to global feature
            global_unified_feature = current_x_global_unified.reshape(current_batch_size, -1)                                              # B * num_keyvoxel * 8 --> B * (num_keyvoxel * 8)
            global_unified_feature = self.unified_global_mapping[str(subj_i)](global_unified_feature)                                      # B * (num_keyvoxel * 8) --> B * 768

            # map to tokens
            unified_global_token_feature = global_unified_feature.unsqueeze(-1)                                                            # B * 768 --> B * 768 * 1
            unified_global_token_feature = self.unified_global_token_mapping[str(subj_i)](unified_global_token_feature)                    # B * 768 * 1 --> B * 768 * num_token
            unified_global_token_feature = unified_global_token_feature.transpose(2, 1)                                                    # B * 768 * num_token --> B * num_token * 768
            
            unified_global_token_feature_list.append(unified_global_token_feature)

        # ------------------------------------------------------------- #

        text_global_token_feature_list = torch.cat(text_global_token_feature_list, dim=0)
        image_global_token_feature_list = torch.cat(image_global_token_feature_list, dim=0)
        unified_global_token_feature_list = torch.cat(unified_global_token_feature_list, dim=0)
        
        # ============================================================= #
        # unified encoder
        # ============================================================= #
        text_token = text_global_token_feature_list
        image_token = image_global_token_feature_list

        # ------------------------------------------------------------ #
        # cross-attention for text token prediction
        pred_text_token = self.TextEncoder(text_token)                                                                                  
        
        # cross-attention for image token prediction
        pred_image_token = self.ImageEncoder(image_token) 
        
        # cross-attention for both token prediction
        uni_token = torch.cat([unified_global_token_feature_list, pred_text_token, pred_image_token], dim=1)

        pred_uni_token = self.UniEncoder(uni_token)

        pred_final_image_token = pred_uni_token[:, :257, :]
        pred_final_text_token = pred_uni_token[:, 257:, :]

        # ------------------------------------------------------------ #
        out_list.append(pred_final_image_token)
        out_list.append(pred_final_text_token)
        out_list.append(pred_image_token)
        out_list.append(pred_text_token)

        return out_list


class UniBrain(nn.Module):
    def __init__(self, args):
        super().__init__()
        """
            the resolution of fMRI is low, fMRI is hard to reconstruct image
            the CLIP text feature is high-level, lacks fine-grained feature
            we first align fMRI with text, and align fMRI with image
            then, we feed the pred image token and text token to a shared Transformer to enable the two modalities to learn from each other
        """
        self.num_keyvoxel = args.num_keyvoxel
        self.num_neighbor = args.num_neighbor
        self.subj_list = args.subj_list
        self.num_map_global_token = args.num_map_global_token                   # map the global feature to several tokens
        self.in_dim = args.in_dim                                               # flattern the fMRI signal, 512 * 32 = 16382
        self.out_dim = args.out_dim                                             # map to CLIP, 768
        self.latent_dim = args.latent_dim
        self.dropout = args.dropout
        self.depth = args.depth                                                 # depth of Transformer
        self.num_subject = len(args.subj_list)
        self.mode_discriminator = args.mode_discriminator                       # linear or non-linear
        self.mode_MutAss = args.mode_MutAss
        self.use_detach = args.use_detach

        # =================================================================feature extractor================================================================= #

        # -------------------------------------------------------------------------------------------------------------------------------------- #
        # text branch
        # -------------------------------------------------------------------------------------------------------------------------------------- #

        # cat the feature from different groups, then perform mapping to global feature
        self.text_group_global_mapping = nn.Linear(self.num_neighbor, self.latent_dim)

        self.text_global_mapping = nn.Linear(self.num_keyvoxel * self.latent_dim, self.out_dim)
        
        # map the global feature to tokens
        self.text_global_token_mapping = nn.Sequential(
            nn.Linear(1, self.num_map_global_token),
            nn.LayerNorm(self.num_map_global_token),
            nn.Dropout(0.5),
        )

        if self.mode_discriminator == 'non-linear':
            self.text_discriminator = nn.Sequential(
                nn.Linear(self.out_dim, 128), 
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_subject),
            )
        elif self.mode_discriminator == 'linear':
            self.text_discriminator = nn.Linear(self.out_dim, self.num_subject)
        else:
            self.text_discriminator = nn.Sequential(
                nn.Linear(self.out_dim, 128), 
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_subject),
            )

        # -------------------------------------------------------------------------------------------------------------------------------------- #
        # image branch
        # -------------------------------------------------------------------------------------------------------------------------------------- #

        # cat the feature from different groups, then perform mapping to global feature
        self.image_group_global_mapping = nn.Linear(self.num_neighbor, self.latent_dim)

        self.image_global_mapping = nn.Linear(self.num_keyvoxel * self.latent_dim, self.out_dim)
        
        # map the global feature to tokens
        self.image_global_token_mapping = nn.Sequential(
            nn.Linear(1, self.num_map_global_token),
            nn.LayerNorm(self.num_map_global_token),
            nn.Dropout(0.5),
        )

        if self.mode_discriminator == 'non-linear':
            self.image_discriminator = nn.Sequential(
                nn.Linear(self.out_dim, 128), 
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_subject),
            )
        elif self.mode_discriminator == 'linear':
            self.image_discriminator = nn.Linear(self.out_dim, self.num_subject)
        else:
             self.image_discriminator = nn.Sequential(
                nn.Linear(self.out_dim, 128), 
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_subject),
            )

        # -------------------------------------------------------------------------------------------------------------------------------------- #
        # unified branch
        # -------------------------------------------------------------------------------------------------------------------------------------- #

        # cat the feature from different groups, then perform mapping to global feature
        self.unified_group_global_mapping = nn.Linear(self.num_neighbor, self.latent_dim)

        self.unified_global_mapping = nn.Linear(self.num_keyvoxel * self.latent_dim, self.out_dim)
        
        # map the global feature to tokens
        self.unified_global_token_mapping = nn.Sequential(
            nn.Linear(1, self.num_map_global_token),
            nn.LayerNorm(self.num_map_global_token),
            nn.Dropout(0.5),
        )

        if self.mode_discriminator == 'non-linear':
            self.unified_discriminator = nn.Sequential(
                nn.Linear(self.out_dim, 128), 
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_subject),
            )
        elif self.mode_discriminator == 'linear':
            self.unified_discriminator = nn.Linear(self.out_dim, self.num_subject)
        else:
             self.unified_discriminator = nn.Sequential(
                nn.Linear(self.out_dim, 128), 
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_subject),
            )

        # =================================================================unified encoder================================================================= #
        # text encoder
        self.TextEncoder = CrossAttention(patch_embed_dim=self.out_dim, hidden_size=self.out_dim, num_latents=77, depth=self.depth)     
        
        # image encoder
        self.ImageEncoder = CrossAttention(patch_embed_dim=self.out_dim, hidden_size=self.out_dim, num_latents=257, depth=self.depth)     

        # unified encoder
        self.UniEncoder = CrossAttention(patch_embed_dim=self.out_dim, hidden_size=self.out_dim, num_latents=257+77, depth=self.depth)  

    def forward(self, x, alpha=0.0):
        out_list = []

        if len(x) == 2 and type(x) is tuple:
            subj_list = x[1].tolist()
            x = x[0]
        else:
            subj_list = self.subj_list

        batch_size, num_keyvoxel, num_neighbor = x.shape
    
        # ============================================================= #
        # extractor
        # ============================================================= #

        # ------------------------------------------------------------- #
        # text feature
        
        # extract feature for each group
        global_text_feature = self.text_group_global_mapping(x)                                                             # B * num_keyvoxel * num_neighbor --> B * num_keyvoxel * 8
        
        # map to global feature
        global_text_feature = global_text_feature.reshape(batch_size, -1)                                                   # B * num_keyvoxel * 8 --> B * (num_keyvoxel * 8)
        global_text_feature = self.text_global_mapping(global_text_feature)                                                 # B * (num_keyvoxel * 8) --> B * 768

        # discriminator
        reverse_global_text_feature = ReverseLayerF.apply(global_text_feature, alpha)
        text_pred_subj = self.text_discriminator(reverse_global_text_feature)                                               # B * num_subj

        # map to tokens
        token_text_feature = global_text_feature.unsqueeze(-1)                                                              # B * 768 --> B * 768 * 1
        token_text_feature = self.text_global_token_mapping(token_text_feature)                                             # B * 768 * 1 --> B * 768 * num_token
        token_text_feature = token_text_feature.transpose(2, 1)                                                             # B * 768 * num_token --> B * num_token * 768

        # ------------------------------------------------------------- #
        # image feature

        # extract feature for each group
        global_image_feature = self.image_group_global_mapping(x)                                                           # B * num_keyvoxel * num_neighbor --> B * num_keyvoxel * 8

        # map to global feature
        global_image_feature = global_image_feature.reshape(batch_size, -1)                                                 # B * num_keyvoxel * 8 --> B * (num_keyvoxel * 8)
        global_image_feature = self.image_global_mapping(global_image_feature)                                              # B * (num_keyvoxel * 8) --> B * 768

        # discriminator
        reverse_global_image_feature = ReverseLayerF.apply(global_image_feature, alpha)
        image_pred_subj = self.image_discriminator(reverse_global_image_feature)                                            # B * num_subj

        # map to tokens
        token_image_feature = global_image_feature.unsqueeze(-1)                                                            # B * 768 --> B * 768 * 1
        token_image_feature = self.image_global_token_mapping(token_image_feature)                                          # B * 768 * 1 --> B * 768 * num_token
        token_image_feature = token_image_feature.transpose(2, 1)                                                           # B * 768 * num_token --> B * num_token * 768
        
        # ------------------------------------------------------------- #
        # unified feature

        # extract feature for each group
        global_unified_feature = self.unified_group_global_mapping(x)                                                       # B * num_keyvoxel * num_neighbor --> B * num_keyvoxel * 8
        
        # map to global feature
        global_unified_feature = global_unified_feature.reshape(batch_size, -1)                                             # B * num_keyvoxel * 8 --> B * (num_keyvoxel * 8)
        global_unified_feature = self.unified_global_mapping(global_unified_feature)                                        # B * (num_keyvoxel * 8) --> B * 768

        # discriminator
        reverse_global_unified_feature = ReverseLayerF.apply(global_unified_feature, alpha)
        unified_pred_subj = self.unified_discriminator(reverse_global_unified_feature)

        # map to tokens
        token_unified_feature = global_unified_feature.unsqueeze(-1)                                                        # B * 768 --> B * 768 * 1
        token_unified_feature = self.unified_global_token_mapping(token_unified_feature)                                    # B * 768 * 1 --> B * 768 * num_token
        token_unified_feature = token_unified_feature.transpose(2, 1)                                                       # B * 768 * num_token --> B * num_token * 768

        # ============================================================= #
        # unified encoder
        # ============================================================= #

        # ------------------------------------------------------------ #
        # cross-attention for text token prediction
        pred_text_token = self.TextEncoder(token_text_feature)                                                              # B * num_text_token * 768                                          
        
        # cross-attention for image token prediction
        pred_image_token = self.ImageEncoder(token_image_feature)                                                           # B * num_image_token * 768      
        
        # cross-attention for both token prediction
        if self.mode_MutAss == 'MutAss':
            if self.use_detach:
                uni_token = torch.cat([token_unified_feature, pred_text_token.detach(), pred_image_token.detach()], dim=1)  # B * (num_token + num_text_token + num_image_token) * 768    
            else:
                uni_token = torch.cat([token_unified_feature, pred_text_token, pred_image_token], dim=1)                    # B * (num_token + num_text_token + num_image_token) * 768    

            pred_uni_token = self.UniEncoder(uni_token)                                                                     # B * (num_text_token + num_image_token) * 768    

            pred_final_image_token = pred_uni_token[:, :257, :]
            pred_final_text_token = pred_uni_token[:, 257:, :]
        
        else:
            # placeholder, useless
            pred_final_text_token = pred_text_token
            pred_final_image_token = pred_image_token       

        # ------------------------------------------------------------ #
        out_list.append(pred_final_image_token)
        out_list.append(pred_final_text_token)
        out_list.append(pred_image_token)
        out_list.append(pred_text_token)

        out_list.append(text_pred_subj)
        out_list.append(image_pred_subj)
        out_list.append(unified_pred_subj)

        return out_list
    

from diffusers.models.vae import Decoder
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, in_dim=15724, h=4096, n_blocks=4, use_cont=False, ups_mode='4x'):
        super().__init__()
        self.lin0 = nn.Sequential(
            nn.Linear(in_dim, h, bias=False),
            nn.LayerNorm(h),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    # @torchsnooper.snoop()
    def forward(self, x, return_transformer_feats=False):
        x = self.lin0(x)
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous())
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)


if __name__ == '__main__':
    input = torch.rand([8, 512, 32]).cuda()

    import argparse

    parser = argparse.ArgumentParser(description='image reconstruction from fMRI signals')

    parser.add_argument('--subj_list', type=int, nargs='+', default=[1, 2, 5, 7], help='which subjects to use for training')
    parser.add_argument("--use_unified_model", action=argparse.BooleanOptionalAction, default=True, help='use a unified model or not')
    parser.add_argument("--use_latent_norm", action=argparse.BooleanOptionalAction, default=True, help='use norm for latent mapping or not')
    parser.add_argument("--use_token_norm", action=argparse.BooleanOptionalAction, default=True, help='use norm for token mapping or not')
    parser.add_argument('--type_norm', type=str, default='BN', help='which type of norm to use')
    parser.add_argument('--num_map_token', type=int, default=512, help='number of kv tokens')
    parser.add_argument('--in_dim', type=int, default=16384, help='dimension of input features')
    parser.add_argument('--out_dim', type=int, default=768, help='dimension of output features, align with CLIP')
    parser.add_argument("--use_img_pos_emb", action=argparse.BooleanOptionalAction, default=True, help='use image positional embedding or not')
    parser.add_argument("--use_text_pos_emb", action=argparse.BooleanOptionalAction, default=True, help='use text positional embedding or not')
    parser.add_argument("--use_token_pos_emb", action=argparse.BooleanOptionalAction, default=True, help='use kv token positional embedding or not')
    parser.add_argument('--num_heads', type=int, default=6, help='number of heads in Transformer')
    parser.add_argument('--depth', type=int, default=6, help='number of layers in Transformer')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--num_keyvoxel', type=int, default=512, help='number of keyvoxels')
    parser.add_argument('--num_neighbor', type=int, default=32, help='number of neighbors for each keyvoxel')
    # rec
    parser.add_argument("--use_rec", action=argparse.BooleanOptionalAction, default=True, help='use reconstruction or not')
    parser.add_argument('--rec_depth', type=int, default=6, help='number of reconstruction Transformer layers')
    parser.add_argument("--use_rec_query_pos_emb", action=argparse.BooleanOptionalAction, default=True, help='use reconstruction query positional embedding or not')
    parser.add_argument("--use_rec_token_pos_emb", action=argparse.BooleanOptionalAction, default=True, help='use reconstruction token positional embedding or not')
    # cyc
    parser.add_argument("--use_cyc", action=argparse.BooleanOptionalAction, default=True, help='use cycle reconstruction or not')

    args = parser.parse_args()

    model = SepMind_MapQ(args).cuda()

    outs = model((input, torch.Tensor(args.subj_list).long().cuda()))
    print(outs)
    for out in outs:
        print(out.shape)
        pdb.set_trace()

