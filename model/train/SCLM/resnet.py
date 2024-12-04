"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as trans
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from torch.optim import lr_scheduler
import torchvision.models as model
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import sklearn
import os
from torch.autograd import Variable


class SupConResNet(nn.Module):
    def __init__(self,):
        super(SupConResNet, self).__init__()
        self.res = model.resnet34(pretrained=True)
        self.fcf = nn.Sequential(nn.Linear(2000, 256),
                                 # nn.BatchNorm1d(512),
                                 # nn.ReLU(), nn.Linear(256, 2),
                                 # nn.Sigmoid())
                                 # active_function
                                 )
        self.res2 = model.resnet34(pretrained=True)
        self.output = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(256),
                                    nn.Linear(256, 2))
    def forward(self, inputdata):
        x0, x1 = inputdata
        x0_ = Variable(x0)
        x1_ = Variable(x1)
        x0 = self.res(x0_)
        x1 = self.res2(x1_)
        h = torch.cat([x0, x1], dim=1)
        h = self.fcf(h)
        output = self.output(h)
        h = F.normalize(h, dim=1)
        return h, output  # 做影像分类的时候，要把注释取消，输出output


class NewRes(nn.Module):
    def __init__(self,):
        super(NewRes, self).__init__()
        self.res = model.resnet34(pretrained=True)
        self.fc1 = nn.Linear(1000,256)
        self.output = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(256),
                                    nn.Linear(256, 2),)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(256, 256)
    def forward(self,x):
        x = self.res(x)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        output = self.output(x)
        x = F.normalize(x, dim=1)
        return x, output


if __name__ == '__main__':
    b = SupConResNet()
    print(b)


