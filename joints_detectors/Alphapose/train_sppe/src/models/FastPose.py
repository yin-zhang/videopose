# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch.nn as nn

from models.layers.DUC import DUC
from models.layers.SE_Resnet import SEResnet

# Import training option
from opt import opt


def createModel():
    return FastPose_SE()


class FastPose_SE(nn.Module):
    conv_dim = 128

    def __init__(self):
        super(FastPose_SE, self).__init__()

        self.preact = SEResnet('resnet101')
        # 2048 * h/32 * w/32

        self.suffle1 = nn.PixelShuffle(2)
        self.duc1 = DUC(512, 1024, upscale_factor=2)
        self.duc2 = DUC(256, 512, upscale_factor=2)
        # 128 * h/8 * w/8

        self.conv_out = nn.Conv2d(
            self.conv_dim, opt.nClasses, kernel_size=3, stride=1, padding=1)
        
        # nn.init.kaiming_normal_(self.conv_out.weight)
        # nn.init.constant_(self.conv_out.bias, 0)

    def forward(self, x):
        # x.shape (2048, w/32, h/32) --> out.shape (512, w/16, h/16)
        out = self.preact(x)
        out = self.suffle1(out)
        out = self.duc1(out)
        out = self.duc2(out)

        out = self.conv_out(out)
        out = out.narrow(1, 0, 17)
        # 128 * h/8 * w/8
        return out
