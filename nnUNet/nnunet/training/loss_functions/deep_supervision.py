#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None, weightT_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.weightT_factors = weightT_factors
        self.loss1 = loss
        self.loss2 = RobustCrossEntropyLoss(reduction='none') 
        
    def forward(self, x, y, out_ds_T=None, Amap=None, U2CT=None, patch_Demb=None, patch_Uemb=None, Attn_map_DL21=None):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors
            
        if self.weightT_factors is None:
            weights_T = [1] * len(out_ds_T)
        else:
            weights_T = self.weightT_factors

        l = self.loss1(x[0], y[0])
        
        for i in range(len(x)-1):
            if weights[i] != 0:
                loss_ds = self.loss1(x[i+1], y[i])
                l += weights[i] * loss_ds
                
                T_CE = self.loss2(out_ds_T[i], y[i][:, 0].long()) * Amap[i]
                loss_T = T_CE.sum() / (Amap[i] == 1).sum()    
                            
                l = l + weights_T[i] * loss_T
                            
        return l