import torch.nn as nn
from torch.nn import functional as F
# import torch
# import numpy as np
import torchvision.models as models
from morpholayers import *
from CSMAM import *
import torch

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3Decoder, DeepLabV3PlusDecoder




class unet_encoder_resnet18(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(unet_encoder_resnet18, self).__init__()
        self.n_channels = n_channels
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'resnet18',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(len(outputs))
        return outputs

class unet_encoder_resnet34(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(unet_encoder_resnet34, self).__init__()
        self.n_channels = n_channels
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'resnet34',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(len(outputs))
        return outputs

class unet_encoder_effb4(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(unet_encoder_effb4, self).__init__()
        self.n_channels = n_channels#[3,48,32,56,160,448]
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'efficientnet-b4',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(outputs[0].shape,outputs[1].shape,outputs[2].shape,outputs[3].shape,outputs[4].shape,outputs[5].shape)
        return outputs


class unetpp_encoder_resnet18(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(unetpp_encoder_resnet18, self).__init__()
        self.n_channels = n_channels
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'resnet18',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(len(outputs))
        return outputs


class unetpp_encoder_resnet34(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(unetpp_encoder_resnet34, self).__init__()
        self.n_channels = n_channels
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'resnet34',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(len(outputs))
        return outputs

class deeplabv3plus_encoder_resnet18(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(deeplabv3plus_encoder_resnet18, self).__init__()
        self.n_channels = n_channels
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'resnet18',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
            output_stride=16,
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(outputs[0].shape,outputs[-2].shape,outputs[-1].shape)
        return outputs

class deeplabv3plus_encoder_resnet34(nn.Module):
    def __init__(self, n_channels, in_channel):
        super(deeplabv3plus_encoder_resnet34, self).__init__()
        self.n_channels = n_channels
        self.in_channel = in_channel
        self.encoder = get_encoder(
            'resnet34',
            in_channels=in_channel,
            depth=len(n_channels),
            weights='imagenet',
            output_stride=16,
        )
    def forward(self, x):
        # print(self.encoder.out_channels)
        outputs = self.encoder(x)
        # print(outputs[0].shape,outputs[-2].shape,outputs[-1].shape)
        return outputs



class unet_decoder_resnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet_decoder_resnet, self).__init__()

        self.decoder = UnetDecoder(
            encoder_channels=(3,64,64,128,256,512),
            decoder_channels=n_channels,
            n_blocks=len(n_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=n_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.decoder(*x)
        outputs = self.segmentation_head(x)
        return outputs

class unet_decoder_effb4(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet_decoder_effb4, self).__init__()

        self.decoder = UnetDecoder(
            encoder_channels=(3,48,32,56,160,448),
            decoder_channels=n_channels,
            n_blocks=len(n_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=n_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.decoder(*x)
        outputs = self.segmentation_head(x)
        return outputs


class unetpp_decoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unetpp_decoder, self).__init__()

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=(3,64,64,128,256,512),
            decoder_channels=n_channels,
            n_blocks=len(n_channels),
            use_batchnorm=True,
            attention_type=None,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=n_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.decoder(*x)
        outputs = self.segmentation_head(x)
        return outputs

class unet_decoder_resnet_CSMAM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet_decoder_resnet_CSMAM, self).__init__()
        down_blocks = []

        down_blocks.append(CSMAM(64, 16,3))
        down_blocks.append(CSMAM(64, 16,3))
        down_blocks.append(CSMAM(128, 16,3))
        down_blocks.append(CSMAM(256, 16, 3))
        down_blocks.append(CSMAM(512, 16, 3))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.decoder = UnetDecoder(
            encoder_channels=(3,64,64,128,256,512),
            decoder_channels=n_channels,
            n_blocks=len(n_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=n_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x_mbam = []
        x_mbam.append(x[0])
        for i in range(1,len(x)):
            x_mbam.append(self.down_blocks[i-1](x[i]))

        # print(x[0].shape,x[1].shape,x[-1].shape)
        x = self.decoder(*x_mbam)
        outputs = self.segmentation_head(x)
        return outputs


class unet_decoder_effb4_CSMAM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet_decoder_effb4_CSMAM, self).__init__()
        down_blocks = []

        down_blocks.append(CSMAM(48, 16,3))
        down_blocks.append(CSMAM(32, 16,3))
        down_blocks.append(CSMAM(56, 16,3))
        down_blocks.append(CSMAM(160, 16, 3))
        down_blocks.append(CSMAM(448, 16, 3))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.decoder = UnetDecoder(
            encoder_channels=(3,48,32,56,160,448),
            decoder_channels=n_channels,
            n_blocks=len(n_channels),
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=n_channels[-1],
            out_channels=n_classes,
            activation=None,
            kernel_size=3,
        )

    def forward(self, x):
        x_mbam = []
        x_mbam.append(x[0])
        for i in range(1,len(x)):
            x_mbam.append(self.down_blocks[i-1](x[i]))

        # print(x[0].shape,x[1].shape,x[-1].shape)
        x = self.decoder(*x_mbam)
        outputs = self.segmentation_head(x)
        return outputs
        
class deeplabv3plus_decoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(deeplabv3plus_decoder, self).__init__()

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=(3,64,64,128,256,512),
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=n_classes,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x):
        x = self.decoder(*x)
        outputs = self.segmentation_head(x)
        return outputs


class EncoderResNet34(nn.Module): ## for classification
    DEPTH = 6
    def __init__(self, n_channels, in_channels):
        super().__init__()
        resnet = torchvision.models.resnet.resnet34(pretrained=True)
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)


    def forward(self, x, with_output_feature_map=False):
        pre_pools = []
        pre_pools.append(x)
        x = self.input_block(x)
        pre_pools.append(x)
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (ResNetUNetEncoder.DEPTH - 1):
                continue
            pre_pools.append(x)
        return pre_pools


class EncoderResNet18(nn.Module):  ## for classification
    DEPTH = 6
    def __init__(self, n_channels, in_channels):
        super().__init__()
        resnet = torchvision.models.resnet.resnet18(pretrained=True)
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)


    def forward(self, x, with_output_feature_map=False):
        pre_pools = []
        pre_pools.append(x)
        x = self.input_block(x)
        pre_pools.append(x)
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (ResNetUNetEncoder.DEPTH - 1):
                continue
            pre_pools.append(x)
        return pre_pools

class EncoderEffb4(nn.Module):  ## for classification
    DEPTH = 6
    def __init__(self, n_channels, in_channels):
        super().__init__()
        efficientnet = torchvision.models.efficientnet_b4(pretrained=True)
        down_blocks = []
        self.input_block = nn.Sequential(*list(efficientnet.children()))[0][:8]

    def forward(self, x, with_output_feature_map=False):
        pre_pools = []
        pre_pools.append(x)
        x = self.input_block(x)
        pre_pools.append(x)

       return pre_pools
       
       
class resnet_classifier(nn.Module):
    def __init__(self, n_classes):
        super(resnet_classifier, self).__init__()
        self.n_classes = n_classes

        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(1024, 128)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(32, n_classes)


    def forward(self, x):

        # x = self.down(x)
        # print(x.shape)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class efficientnet_classifier(nn.Module):
    def __init__(self, n_classes):
        super(efficientnet_classifier, self).__init__()
        self.n_classes = n_classes
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(448, 128)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(32, n_classes)


    def forward(self, x):


        # print(x.shape)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class resnet_classifier_CA(nn.Module):
    def __init__(self, n_classes):
        super(resnet_classifier_CA, self).__init__()
        self.n_classes = n_classes
        self.down_blocks_bridge =  CSMAM(512,16,3,no_spatial = True)
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(1024, 128)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(32, n_classes)


        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(512, 128)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(32, n_classes)
    def forward(self, x):

        x = self.down_blocks_bridge(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class efficientnet_classifier_CA(nn.Module):
    def __init__(self, n_classes):
        super(efficientnet_classifier_CA, self).__init__()
        self.n_classes = n_classes
        self.down_blocks_bridge =  CSMAM(448,16,3,no_spatial = True)
        self.pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(448, 128)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(128, 32)
        self.drop2 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(32, n_classes)


    def forward(self, x):


        # print(x.shape)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x
