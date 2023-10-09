import numba as nb
from audioop import bias
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from nnunet.network_architecture.neural_network import SegmentationNetwork
from einops import rearrange
from .vit_2D import *


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        
        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs[
                'p'] > 0:
        
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        
        self.lrelu = nonlin(**nonlin_kwargs) if nonlin_kwargs != None else nonlin() 

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class DownOrUpSample(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels,
                 conv_op, conv_kwargs,
                 norm_op=nn.InstanceNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, nonlinbasic_block=ConvDropoutNormNonlin):
        super(DownOrUpSample, self).__init__()
        self.blocks = nonlinbasic_block(input_feature_channels, output_feature_channels, conv_op, conv_kwargs,
                                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        nonlin, nonlin_kwargs)

    def forward(self, x):
        return self.blocks(x)

class DeepSupervision(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.proj = nn.Conv2d(
            dim, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x

def uncertainty_map(x,num_classes):
    prob = F.softmax(x,dim=1)     
    log_prob = torch.log2(prob + 1e-6)
    entropy = -1 * torch.sum(prob * log_prob, dim=1) / math.log2(num_classes)
    one = torch.ones_like(entropy, dtype=torch.int8)
    zero = torch.zeros_like(entropy, dtype=torch.int8)
    entropy = torch.where(entropy>=0.001,one,zero)
    return entropy

@nb.jit()
def UnoB(tar,amap,S):
    neibor = np.ones((S,S))
    pro = np.ones_like(amap)
    s = S//2
    for b in range(amap.shape[0]):
        for i in range(s,amap.shape[1]-s):
            for j in range(s,amap.shape[2]-s):
                if tar[b,i,j]!=0:
                    if S == 3:
                        pro[b,i,j] = 0 if 4*tar[b,i,j] != (tar[b,i+1,j]+tar[b,i-1,j]+tar[b,i,j+1]+tar[b,i,j-1]) else 1
                    else:
                        pro[b,i,j] = 0 if S*S*tar[b,i,j] != np.sum(tar[b,i-s:i+s+1,j-s:j+s+1] * neibor) else 1
                        
    Amap_pro = amap * pro
    return Amap_pro

class BasicLayer(nn.Module):

    def __init__(self, num_stage, num_pool, base_num_features, input_resolution=None, bound_width=None, 
                 dmodel=None, depth=None, num_heads=None, add_Map=True, patch_size=None, dim_head=None, 
                 image_channels=1, num_conv_per_stage=2, conv_op=None, norm_op=None, norm_op_kwargs=None, 
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, basic_block=ConvDropoutNormNonlin, 
                 max_num_features=None, down_or_upsample=None, feat_map_mul_on_downscale=2,
                 num_classes=None, is_encoder=True, use_checkpoint=False):

        super().__init__()
        self.num_stage = num_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder
        self.num_classes = num_classes
        self.bound_size = bound_width
        dim = min((base_num_features * feat_map_mul_on_downscale **
                       num_stage), max_num_features)
        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        if num_stage == 0 and is_encoder:
            input_features = image_channels
        elif not is_encoder and num_stage < num_pool:
            input_features = 2*dim
        else:
            input_features = dim
        
        
        # self.depth = depth
        conv_kwargs['kernel_size'] = conv_kernel_sizes[num_stage]
        conv_kwargs['padding'] = conv_pad_sizes[num_stage]

        self.input_du_channels = dim
        self.output_du_channels = min(int(base_num_features * feat_map_mul_on_downscale ** (num_stage+1 if is_encoder else num_stage-1)),
                                      max_num_features)
        self.conv_blocks = nn.Sequential(
            *([basic_block(input_features, dim, conv_op,
                           conv_kwargs,
                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                           nonlin, nonlin_kwargs)] +
              [basic_block(dim, dim, conv_op,
                           conv_kwargs,
                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                           nonlin, nonlin_kwargs) for _ in range(num_conv_per_stage - 1)]))

        # build blocks
        if not is_encoder and num_stage in [0,1,2,3,4]:
            self.vit_blocks = Vision_Transformer(dim=dim, dmodel=dmodel, input_resolution=input_resolution, 
                                num_heads=num_heads, patch_size=patch_size, dropout=0.1,
                                in_depth=depth, add_Map=add_Map, dim_head=dim_head)
            
        # patch merging layer
        if down_or_upsample is not None:
            dowm_stage = num_stage-1 if not is_encoder else num_stage
            self.down_or_upsample = nn.Sequential(down_or_upsample(self.input_du_channels, self.output_du_channels, pool_op_kernel_sizes[dowm_stage],
                                                                   pool_op_kernel_sizes[dowm_stage], bias=False),
                                                  norm_op(self.output_du_channels, **norm_op_kwargs)
                                                  )
        else:
            self.down_or_upsample = None
        
        if not is_encoder:
            self.deep_supervision = DeepSupervision(dim, num_classes)
            self.deep_supervision_T = DeepSupervision(dim, num_classes) if (num_stage in [0,1,2,3,4]) else None
            self.deep_supervision_out = DeepSupervision(dim, num_classes) if num_stage==0 else None
        else:
            self.deep_supervision = None
            self.deep_supervision_T = None
            self.deep_supervision_out = None
        
    def forward(self, x, skip, tar):
        if not self.is_encoder and self.num_stage < self.num_pool:
            x = torch.cat((x, skip), dim=1)
        
        x = self.conv_blocks(x)
        if self.deep_supervision is not None:
            ds = self.deep_supervision(x)
        
        if not self.is_encoder and self.num_stage in [0,1,2,3,4]:
            s = x.detach()

            if tar is None:
                tar = torch.argmax(ds, dim=1, keepdim=True)
            A_map = uncertainty_map(ds.clone().detach(), self.num_classes)
            pro = UnoB(tar[:, 0].long().detach().cpu().numpy(),A_map.cpu().numpy(),self.bound_size)
            A_map = torch.tensor(pro).cuda()                 
            
            if self.use_checkpoint:
                s = checkpoint.checkpoint(self.vit_blocks, s, A_map)
            else:
                s = self.vit_blocks(s, A_map)
            if self.deep_supervision_T is not None:
                ds_T = self.deep_supervision_T(s)
            
            x = x + s
                   
        if self.down_or_upsample is not None:
            du = self.down_or_upsample(x)
        if self.deep_supervision_out is not None:
            final = self.deep_supervision_out(x)   

        if self.is_encoder: 
            return x, du, None, None
        elif self.down_or_upsample is not None:
            if self.num_stage != 5:
                return du, ds, A_map, ds_T
            else:
                return du, ds, None, None
        elif self.down_or_upsample is None:
            return final, ds, A_map, ds_T

class UCTNet_2D(SegmentationNetwork):
    def __init__(self, img_size, base_num_features, num_classes, image_channels=1, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2,  pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 deep_supervision=True, max_num_features=None, bound_width=None,
                 dmodels=None, depths=None, num_heads=None, patch_size=None, dim_head=None, add_Map=True,
                 dropout_p=0.1, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.conv_op = nn.Conv2d
        norm_op = nn.InstanceNorm2d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout3d 
        dropout_op_kwargs = {'p': dropout_p, 'inplace': True}
        nonlin = nn.GELU 
        nonlin_kwargs = None
       
        self.do_ds = deep_supervision
        self.num_pool = len(pool_op_kernel_sizes)
        conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        # build layers
        self.down_layers = nn.ModuleList()
        for i_layer in range(self.num_pool):  # 0,1,2,3,4
            layer = BasicLayer(num_stage=i_layer, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               down_or_upsample=nn.Conv2d,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               num_classes=self.num_classes,
                               use_checkpoint=use_checkpoint,
                               is_encoder=True)
            self.down_layers.append(layer)
        self.up_layers = nn.ModuleList()
        for i_layer in range(self.num_pool+1)[::-1]: # 5,4,3,2,1,0
            layer = BasicLayer(num_stage=i_layer, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                   img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)), 
                               bound_width=bound_width[i_layer],
                               dmodel=dmodels[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               patch_size=patch_size[i_layer],
                               dim_head=dim_head,
                               add_Map=add_Map,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               down_or_upsample=nn.ConvTranspose2d if (
                                   i_layer > 0) else None,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               num_classes=self.num_classes,
                               use_checkpoint=use_checkpoint,
                               is_encoder=False)
            self.up_layers.append(layer)
        self.apply(self._InitWeights)
    def _InitWeights(self,module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=.02)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x, tar, is_infer):
        x_skip = []
        for inx, layer in enumerate(self.down_layers):
            s, x, _, _ = layer(x, None, None)
            x_skip.append(s)
            
        out = []
        out_ds_T, out_Amap = [], []
        for inx, layer in enumerate(self.up_layers):
            if is_infer:
                x, ds, A_map, ds_T = layer(x, x_skip[self.num_pool-inx], None) if inx > 0 else layer(x, None, None)
            else:
                x, ds, A_map, ds_T = layer(x, x_skip[self.num_pool-inx], tar[self.num_pool-inx]) if inx > 0 else layer(x, None, None)
            if inx > 0:
                out.append(ds)
                out_ds_T.append(ds_T)
                out_Amap.append(A_map)
            if inx == 5:
                out.append(x)
                
        if self.do_ds:
            return out[::-1], out_ds_T[::-1], out_Amap[::-1]
        else:
            return out[-1]
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}
