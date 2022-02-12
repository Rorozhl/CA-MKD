from __future__ import print_function
from numpy import append
from numpy.core.fromnumeric import transpose

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature map layer)"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.s_H * 4 == self.t_H:
            x = F.interpolate(x, size=(self.t_H, self.t_H), mode='bilinear')
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)
        

class Regress(nn.Module):
    """Simple Linear Regression for FitNet (feature vector layer)"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x
        
class CalWeight(nn.Module):
    def __init__(self, feat_s, feat_t_list, opt):
        super(CalWeight, self).__init__()

        self.opt = opt
        # student和teacher都用最后一层
        s_channel = feat_s.shape[1]
        for i in range(len(feat_t_list)):
            t_channel = feat_t_list[i].shape[1]
            setattr(self, 'embed'+str(i), Embed(s_channel, t_channel, self.opt.factor, self.opt.convs))


    def forward(self, feat_s, feat_t_list, model_t_list=None):
        tmp_model = [model_t.distill_seq() for model_t in model_t_list]
        trans_feat_s_list = []
        output_feat_t_list = []
        s_H = feat_s.shape[2]
        for i, mid_feat_t in enumerate(feat_t_list):
            t_H = mid_feat_t.shape[2]
            if s_H >= t_H:
                feat_s = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            else:
                feat_s = F.interpolate(feat_s, size=(t_H, t_H), mode='bilinear')
            trans_feat_s = getattr(self, 'embed'+str(i))(feat_s)
            trans_feat_s_list.append(trans_feat_s)

            output_feat_t = tmp_model[i][-1](trans_feat_s)
            output_feat_t_list.append(output_feat_t)
        return trans_feat_s_list, output_feat_t_list




class AAEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(AAEmbed, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            # conv1x1(num_input_channels, self.num_mid_channel),
            # nn.BatchNorm2d(self.num_mid_channel),
            # nn.ReLU(inplace=True),
            # conv3x3(self.num_mid_channel, self.num_mid_channel),
            # nn.BatchNorm2d(self.num_mid_channel),
            # nn.ReLU(inplace=True),
            # conv1x1(self.num_mid_channel, num_target_channels),
            conv1x1(num_input_channels, num_target_channels),
            nn.BatchNorm2d(num_target_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
        
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, factor=2, convs=False):
        super(Embed, self).__init__()
        self.convs = convs
        if self.convs:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_in//factor, kernel_size=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in//factor, dim_in//factor, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True), 
                nn.Conv2d(dim_in//factor, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)              
            )
        else:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True) 
            )


    def forward(self, x):
        x = self.transfer(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
