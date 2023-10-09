from visualizer import get_local
import time
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
from .swin_3D import *
from .vit_3D import *

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
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
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
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
        self.proj = nn.Conv3d(
            dim, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x

def uncertainty_map(x,num_classes):
    prob = F.softmax(x,dim=1)     
    log_prob = torch.log2(prob + 1e-10)
    entropy = -1 * torch.sum(prob * log_prob, dim=1) / math.log2(num_classes)
    one = torch.ones_like(entropy)
    zero = torch.zeros_like(entropy)
    entropy = torch.where(entropy>=0.001,one,zero)
    print('entropy!=0',(entropy[0] == 1).sum().item())
    print('entropy==0',(entropy[0] < 1).sum().item())
    return entropy

class BasicLayer(nn.Module):

    def __init__(self, num_stage, num_only_conv_stage, num_pool, base_num_features,image_channels=1, num_conv_per_stage=2, conv_op=None,
                 norm_op=None, norm_op_kwargs=None,dropout_op=None, dropout_op_kwargs=None,nonlin=None, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None, basic_block=ConvDropoutNormNonlin, 
                 max_num_features=None,down_or_upsample=None, feat_map_mul_on_downscale=2,
                 num_classes=None, is_encoder=True,use_checkpoint=False):

        super().__init__()
        self.num_stage = num_stage
        self.num_only_conv_stage = num_only_conv_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder
        self.num_classes = num_classes
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
        else:
            self.deep_supervision = None
        
    def forward(self, x, skip):
        if not self.is_encoder and self.num_stage < self.num_pool:
            x = torch.cat((x, skip), dim=1)
        
        x = self.conv_blocks(x)
        print("before ds x'size",x.size())
        if self.deep_supervision is not None:
            ds = self.deep_supervision(x)
            
        if self.down_or_upsample is not None:
            du = self.down_or_upsample(x)
                

        if self.is_encoder: 
            return x, du
        elif self.down_or_upsample is not None:
            return du, ds
        elif self.down_or_upsample is None:
            return None, ds

class MYTrans_m2(SegmentationNetwork):
    def __init__(self, base_num_features, num_classes, image_channels=1, num_only_conv_stage=2, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2,  pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, deep_supervision=True, max_num_features=None, dropout_p=0.1,
                  use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
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
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               down_or_upsample=nn.Conv3d,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               num_classes=self.num_classes,
                               use_checkpoint=use_checkpoint,
                               is_encoder=True)
            self.down_layers.append(layer)
        self.up_layers = nn.ModuleList()
        for i_layer in range(self.num_pool+1)[::-1]: # 5,4,3,2,1,0
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=num_only_conv_stage, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               image_channels=image_channels, num_conv_per_stage=num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes, pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               down_or_upsample=nn.ConvTranspose3d if (
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

    def forward(self, x, tar, is_infer):#
        x_skip = []
        for inx, layer in enumerate(self.down_layers):
            s, x = layer(x, None)
            x_skip.append(s)
            
        out = []
        for inx, layer in enumerate(self.up_layers):
            x, ds = layer(x, x_skip[self.num_pool-inx]) if inx > 0 else layer(x, None)
            if inx > 0:
                out.append(ds)
        if self.do_ds:
            return out[::-1]
        else:
            return out[-1], out[-3], out[-3], out[-3][:,0]
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}
