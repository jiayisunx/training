# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
from torch import nn
import numpy as np

if os.environ.get('TRAIN') == "1":
    class FrozenBatchNorm2d(nn.Module):
        """
        BatchNorm2d where the batch statistics and the affine parameters
        are fixed
        """
        def __init__(self, n):
            super(FrozenBatchNorm2d, self).__init__()
            self.register_buffer("weight", torch.ones(n))
            self.register_buffer("bias", torch.zeros(n))
            self.register_buffer("running_mean", torch.zeros(n))
            self.register_buffer("running_var", torch.ones(n))

        def forward(self, x):
            scale = self.weight * self.running_var.rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            if x.is_mkldnn and x.dtype == torch.bfloat16:
                return (x.to_dense(torch.float) * scale + bias).to_mkldnn(torch.bfloat16)
            elif x.is_mkldnn:
                return (x.to_dense() * scale + bias).to_mkldnn()
            else:
                return x * scale + bias
else:
    class FrozenBatchNorm2d(nn.BatchNorm2d):
        def forward(self, x):
            return super(FrozenBatchNorm2d, self).forward(x)
