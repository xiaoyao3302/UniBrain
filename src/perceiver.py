"""
Copied from
https://github.com/lucidrains/flamingo-pytorch/blob/main/flamingo_pytorch/flamingo_pytorch.py
"""

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops_exts import rearrange_many
import pdb

def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )


class MLPs(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = dim * mult
        self.norm = nn.LayerNorm(dim)
        self.mlp1 = nn.Linear(dim, inner_dim, bias = False)
        self.act = nn.GELU()
        self.mlp2 = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads                # 96 * 16

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        # x: 64, 1, 256, 1024
        # latents: 64, 1, 256, 1024
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads              # 64, 1, 16

        q = self.to_q(latents)                          # 64, 1, 256, 1536

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to

        kv_input = torch.cat((x, latents), dim = -2)        # 在token的维度拼接  64, 1, 256, 1536 + 64, 1, 256, 1536 = 64, 1, 512, 1536
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)      # k, v are both with the same dimension of 64, 1, 512, 1536

        # k, v = self.to_kv(x).chunk(2, dim = -1)      # k, v are both with the same dimension of 64, 1, 512, 1536

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)      # divide into multiple heads

        # q: 64, 16, 1, 256, 96
        # k: 64, 16, 1, 512, 96
        # v: 64, 16, 1, 512, 96

        q = q * self.scale      # 0.10206200XXX

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)              # 64, 16, 1, 256, 512

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()         # 尝试一下这个逻辑的作用

        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)            # 64, 16, 1, 256, 96
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)         # 64, 1, 256, 1536  
        return self.to_out(out)                                         # 64, 1, 256, 1024

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_media_embeds = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))                      # 256, 1024 这里就是初始化了一系列的query
        self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))        # 1, 1, 1024

        # plan A, plan B: self-attention or cross-attention
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # # plan C: MLP with equal size
        # self.layers = nn.ModuleList([])
        # for _ in range(depth*2):
        #     self.layers.append(MLPs(dim = dim, mult = ff_mult))

        # # plan D: MLP with equal layer
        # self.layers = nn.ModuleList([])
        # for _ in range(depth):
        #     self.layers.append(MLPs(dim = dim, mult = ff_mult))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # B, 256, 1024
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')        # B, 1, 261, 1024

        times = x.shape[1]
        x = x + self.media_pos_emb[:times]

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])        #  B, 1, 256, 1024

        # plan A: in raw paper, embeddings as query
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents                                                # B, 1, 256, 1024
            latents = ff(latents) + latents

        # # plan B: self attention
        # for attn, ff in self.layers:
        #     x = attn(x, x) + x                                                # B, 1, 256, 1024
        #     x = ff(x) + x
        # latents = x[:, :, :256, :]

        # # plan C, plan D: MLPs
        # for mlps in self.layers:
        #     x = mlps(x) + x                                                # B, 1, 256, 1024
        # latents = x[:, :, :256, :]

        # dont't move
        res = self.norm(latents)

        if res.ndim == 4:
            res = res.squeeze(1)                                                                # B, 256, 1024

        return res