import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from morpholayers import *


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        # print(torch.unique(x_out))
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        # print(torch.unique(x_out))
        return x_out

class MorphChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(MorphChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.mssm = MSMCM_single(gate_channels,7)
    def forward(self, x):
        x_morph = self.mssm(x)
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x_morph * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DEChannelPool(nn.Module):
    def __init__(self, in_channel,kernel):
        super(DEChannelPool, self).__init__()
        n = in_channel

        self.ers1 = Erosion2d(in_channel, kernel, soft_max=True,dilation = 1)
        self.ers2 = Erosion2d(in_channel, kernel, soft_max=True,dilation = 2)
        self.ers3 = Erosion2d(in_channel, kernel, soft_max=True,dilation = 3)

        self.dil1 = Dilation2d(in_channel, kernel, soft_max=True,dilation = 1)
        self.dil2 = Dilation2d(in_channel, kernel, soft_max=True,dilation = 2)
        self.dil3 = Dilation2d(in_channel, kernel, soft_max=True,dilation = 3)

        self.bne1 = nn.BatchNorm2d(in_channel)
        self.bne2 = nn.BatchNorm2d(in_channel)
        self.bne3 = nn.BatchNorm2d(in_channel)

        self.bnd1 = nn.BatchNorm2d(in_channel)
        self.bnd2 = nn.BatchNorm2d(in_channel)
        self.bnd3 = nn.BatchNorm2d(in_channel)

        self.conve = nn.Conv2d(in_channel*3, 1, kernel_size=1, padding=0, dilation=1, bias=False)
        self.convd = nn.Conv2d(in_channel*3, 1, kernel_size=1, padding=0, dilation=1, bias=False)
        self.bne = nn.BatchNorm2d(1)
        self.bnd = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        batch, _, h, w = x.size()
        xe1 = self.bne1(self.ers1(x))
        xe2 = self.bne2(self.ers2(x))
        xe3 = self.bne3(self.ers3(x))

        xd1 = self.bnd1(self.dil1(x))
        xd2 = self.bnd2(self.dil2(x))
        xd3 = self.bnd3(self.dil3(x))

        xe = torch.cat((xe1, xe2, xe3), 1)
        xe = self.relu(self.bne(self.conve(xe)))
        xd = torch.cat((xd1, xd2, xd3), 1)
        xd = self.relu(self.bnd(self.convd(xd)))

        x = torch.cat((xe, xd), 1)

        return x  


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
        
 
class DESpatialGate(nn.Module):
    def __init__(self,n_chan,kernel):
        super(DESpatialGate, self).__init__()
        kernel_size = 3
        self.compress = DEChannelPool(n_chan,kernel)
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class CSMAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, kernel=3,pool_types=['avg', 'max'], no_spatial=False):
        super(CSMAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = DESpatialGate(gate_channels, kernel)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



