from models_multitask import *
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_encoder(encoder_name, channels, in_chan):
    if encoder_name == 'unet_encoder_resnet18':
        return unet_encoder_resnet18(channels, in_chan)
    elif encoder_name == 'unet_encoder_resnet34':
        return unet_encoder_resnet34(channels, in_chan)
    elif encoder_name == 'unet_encoder_effb4':
        return unet_encoder_effb4(channels, in_chan)
    elif encoder_name == 'unetpp_encoder_resnet18':
        return unetpp_encoder_resnet18(channels, in_chan)
    elif encoder_name == 'unetpp_encoder_resnet34':
        return unetpp_encoder_resnet34(channels, in_chan)
    elif encoder_name == 'deeplabv3plus_encoder_resnet18':
        return deeplabv3plus_encoder_resnet18(channels, in_chan)
    elif encoder_name == 'deeplabv3plus_encoder_resnet34':
        return deeplabv3plus_encoder_resnet18(channels, in_chan)


def get_decoder(decoder_name, channels, n_class):
    if decoder_name == 'unet_decoder_resnet':
        return unet_decoder_resnet(channels, n_class)
    elif decoder_name == 'unet_decoder_effb4':
        return unet_decoder_effb4(channels, n_class)
    elif decoder_name == 'unetpp_decoder':
        return unetpp_decoder(channels, n_class)
    elif decoder_name == 'unet_decoder_resnet_CSMAM':
        return unet_decoder_resnet_CSMAM(channels, n_class)
    elif decoder_name == 'unet_decoder_effb4_CSMAM':
        return unet_decoder_effb4_CSMAM(channels, n_class)
    elif decoder_name == 'deeplabv3plus_decoder':
        return deeplabv3plus_decoder(channels, n_class)


def get_classifier(classifier_name, n_class):
    if classifier_name == 'resnet_classifier':
        return resnet_classifier(n_class)
    elif classifier_name == 'resnet_classifier_CA':
        return resnet_classifier_CA(n_class)
    elif classifier_name == 'efficientnet_classifier':
        return efficientnet_classifier(n_class)
    elif classifier_name == 'efficientnet_classifier_CA':
        return efficientnet_classifier_CA(n_class)

        


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, classifier_name, inchan, channels, numclass,mode):
        super(MultiTaskModel,self).__init__()
        self.encoder_name = encoder_name
        self.mode = mode
        self.inchan = inchan
        self.channels = channels
        self.numclass = numclass

        self.encoder = get_encoder(encoder_name,channels,inchan)    
        if mode == 'classification':
            self.classifier_name = classifier_name
            self.classHead = get_classifier(classifier_name, numclass)
        elif mode == 'segmentation':
            self.decoder_name = decoder_name
            self.segHead = get_decoder(decoder_name,channels,1)
        elif mode == 'multitask':
            self.classifier_name = classifier_name
            self.decoder_name = decoder_name
            self.classHead = get_classifier(classifier_name, numclass)
            self.segHead = get_decoder(decoder_name,channels,1)
        
    def forward(self,x):

        x = self.encoder(x)
        if self.mode == 'classification':
            x_class = self.classHead(x[-1])
            return x_class
        elif self.mode == 'segmentation':
            x_seg = self.segHead(x)
            return x_seg
        elif self.mode == 'multitask':
            # print(x[-1].shape)
            x_class = self.classHead(x[-1])
            x_seg = self.segHead(x)
            return x, x_seg, x_class
