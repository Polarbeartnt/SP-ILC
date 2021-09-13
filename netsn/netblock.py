import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from PIL import Image

#-------------------------------------------------#
#   Convolution Block
#   CONV+BatchNorm+LeakyReLU
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   Structure Block of CSPdarknet53-tiny
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels//2, 3)

        self.conv2 = BasicConv(out_channels//4, out_channels//4, 3)
        self.conv3 = BasicConv(out_channels//4, out_channels//4, 3)

        self.conv4 = BasicConv(out_channels//2, out_channels//2, 1)

    def forward(self, x):
        x = self.conv1(x)
        route = x
        
        c = self.out_channels
        x = torch.split(x, c//4, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x,route1], dim = 1) 
        x = self.conv4(x)
        feat = x

        x = torch.cat([route, x], dim=1)
        return x,feat


class FcLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.ac = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.ac(x)
        return x

#---------------------------------------------------#
#   Conv + Upsample
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 3),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   Yolo head for v4
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
