import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

np.set_printoptions(threshold=1000)

from torchvision.ops import roi_align, nms


class Map_reshape(nn.Module):   
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.p1, self.p2, self.p3,= patch_size[0], patch_size[1], patch_size[2]
        self.num_heads = num_heads

    def forward(self, map): 
        map = rearrange(map, 'b (s p1) (h p2) (w p3) -> b (s h w) (p1 p2 p3)', p1=self.p1, p2=self.p2, p3=self.p3)
        map = map.max(-1)[0]
        map_attn = map.unsqueeze(2).repeat(1,1,map.shape[-1]).unsqueeze(1).repeat(1,self.num_heads,1,1)
        return map_attn

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, map_attn, add_Map, **kwargs):
        return self.fn(self.norm(x), map_attn, add_Map, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, map_attn, add_Map):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,  dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, map_attn, add_Map):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
                
        attn = self.attend(dots)    # softmax
        if add_Map and (map_attn is not None):
            attn = map_attn * attn 
    
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=num_heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x, map_attn, add_Map):
        for attn, ff in self.layers:
            att_out = attn(x,map_attn,add_Map)
            x = att_out + x
            x = ff(x,None,None) + x
        return x

class Vision_Transformer(nn.Module):
    def __init__(self, dim, dmodel, input_resolution, num_heads, patch_size=[1,1,1], dropout=0.1, emb_dropout=0.1, in_depth=1, mlp_dim=3072, dim_head=64, add_Map=True):
        super().__init__()
        
        self.dim = dim
        self.dmodel = dmodel
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_dim = self.dmodel*4
        self.add_Map = add_Map
        
        S, H, W = self.input_resolution
        assert S % patch_size[0] == 0 and H % patch_size[1] == 0 and W % patch_size[2] == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (S // patch_size[0]) * (H // patch_size[1]) * (W // patch_size[2]) 
        patch_dim = patch_size[0] * patch_size[1] * patch_size[2] * dim
        
        self.map_reshape = Map_reshape(patch_size, self.num_heads)
        
        self.patch_embedding = nn.Sequential(
            Rearrange('b c (s p1) (h p2) (w p3) -> b (s h w) (p1 p2 p3 c)', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]),
            nn.Linear(patch_dim, self.dmodel),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, in_depth, num_heads, dim_head, self.mlp_dim, dropout)
        self.recover_patch_embedding = nn.Sequential(
            nn.Linear(self.dmodel, patch_dim),
            Rearrange('b (s h w) (p1 p2 p3 c) -> b c (s p1) (h p2) (w p3)', 
                    h=(H//patch_size[1]),w=(W//patch_size[2]), p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]),
        )
        
    def forward(self, x, A_map):
        s, h, w = self.input_resolution
        B, C, S, H, W = x.shape
        assert S == s and H == h and W == w, "input feature has wrong size"
        
        x = self.patch_embedding(x)  
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        if A_map is not None:
            map_attn = self.map_reshape(A_map)
        else:
            map_attn = None
        
        x = self.transformer(x, map_attn, self.add_Map)
        vit_out = self.recover_patch_embedding(x)
        
        return vit_out
