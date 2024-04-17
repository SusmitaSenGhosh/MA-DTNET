import numpy 
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
import ot
import torch
from ot.gromov import gromov_wasserstein2, gromov_wasserstein
import torch.nn as nn


def Dice(output,target,weight=None, eps=1e-5):
    target = target.float()
    if weight is None:
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
    else:

        # sum_dims = list(range(1, output.dim()))
        #
        # num = 2 * torch.sum(weight * output * target, dim=sum_dims)
        # den= torch.sum(weight * (output + target), dim=sum_dims) + eps
        num = 2 * (weight * output * target).sum()
        den = (weight*output).sum() + (weight*target).sum() + eps
    return num/den




class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
