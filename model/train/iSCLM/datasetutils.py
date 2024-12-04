from __future__ import print_function
import os
import sys
import argparse
import time
import math
import SimpleITK as sitk
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
# from resnet_big import LinearClassifier # ImgEncoder,
from losses import SupConLoss
from torch.utils.data import Dataset
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
import cv2
import matplotlib.pyplot as plt
import sklearn
import os


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, rootpath, csvpath, niiname = 'tumor_resam.nii.gz',test=True):
        # self.csv = pd.read_excel(csvpath,)  # encoding='GBK')
        self.names = csvpath
        self.test = test
        print('self.test..................',self.test)
        # self.dic = self.generate_dic()
        self.imgpath = [os.path.join(rootpath, str(i), niiname) for i in self.names]
        # self.label = [self.dic[i] for i in self.names]
        # self.transform_test = trans.Compose([
        #     # transforms.Resize((256, 256)),
        #     trans.ToTensor(),
        #     trans.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
        self.trans_ct = trans.Compose([
            trans.ToPILImage(),
            trans.Resize((224, 224)),
            trans.RandomHorizontalFlip(),  # random test
            trans.ToTensor(),
            # trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.trans_cttest = trans.Compose([
        #     trans.ToPILImage(),
        #     trans.Resize((224, 224)),
        #     trans.RandomHorizontalFlip(),    # 这里加入了随机增强测试，测试时随机反转
        #     trans.ToTensor(),
        #     # trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    def getnames(self):
        return self.names

    # def generate_dic(self):
    #     dic_ct_label = {}
    #     for i in range(len(self.names)):
    #         dic_ct_label[self.names[i]] = self.csv['label'][i]
    #     return dic_ct_label

    def __getitem__(self, index):
        imgpath = self.imgpath[index]
        img = sitk.ReadImage(imgpath)
        img_3C = sitk.ReadImage(imgpath.replace('.nii','3ceng.nii'))
        # imgnp_ori= sitk.GetArrayFromImage(img)[0,:,:]
        imgnp_ori = sitk.GetArrayFromImage(img)
        img_3Cnp = sitk.GetArrayFromImage(img_3C)
        # print('np最大灰度值', int(np.max(imgnp_ori)))
        if int(np.max(imgnp_ori)) == 0:
            imgnp_ori[imgnp_ori==float(0)]=255
        # 求连通域
        imgnp = cv2.threshold(imgnp_ori, 20, 255, cv2.THRESH_BINARY)[1]
        imgnp = np.asarray(imgnp,dtype=np.uint8)
        image_, cnts_, = cv2.findContours(imgnp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 最小外接矩形
        x, y, w, h = cv2.boundingRect(image_[0])
        # print('x,y,w,h,len(image_):',x,y,w,h,len(image_))
        if len(image_) ==2 :
            x2, y2, w2, h2 = cv2.boundingRect(image_[1])
            x, y, w, h = min(x,x2), min(y,y2), max(w2+x2,w+x)-min(x,x2), max(y2+h2,y+h)-min(y,y2)
        # 截取图像
        width = 10
        # newimgnp = imgnp_ori[y - width:int(y + h) + width, x - width:int(x + w) + width]
        img_3Cnp_crop = img_3Cnp[ :, y - width:int(y + h) + width, x - width:int(x + w) + width]
        if len(image_) >= 3:
            print('warning!', len(image_), imgpath)
            # newimgnp = imgnp_ori
            img_3Cnp_crop = img_3Cnp
        # imgnp = np.stack([newimgnp, newimgnp, newimgnp], axis=0)

        imgnp = img_3Cnp_crop
        # plt.imshow(newimgnp)
        # plt.show()
        # for i in range(0,3):
        #     plt.imshow(imgnp[i,:,:])
        #     plt.show()
        max_, min_ = np.max(imgnp), np.min(imgnp)
        if max_==min_==0:
            imgnp = imgnp
        else:
            imgnp = (imgnp-min_)/(max_-min_)
        imgnp = imgnp.astype(np.float32)
        imgnp = torch.from_numpy(imgnp)
        imgtf = self.trans_ct(imgnp)
        # if not self.test:
        #     imgtf = self.trans_ct(imgnp)
        # else:
        #     imgtf = self.trans_ct(imgnp)
        # print(imgtf.size())  # 3,192,192
        # label = self.label[index]
        return imgtf,

    def __len__(self):
        return len(self.imgpath)


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    opt.warmup_from = 0.01
    opt.warm_epochs = 5
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate_cos(optimizer, epoch):
    lr = 0.005
    lr_decay_rate = 0.1
    all_epoch = 200
    eta_min = lr * (lr_decay_rate ** 3)

    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / all_epoch)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_warm_cos(optimizer, current_epoch, max_epoch, lr_max= 0.005):
    lr_decay_rate = 0.1
    lr_min = lr_max * (lr_decay_rate ** 3)
    warmup_epoch = 20

    #print(lr_max , current_epoch , warmup_epoch)
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + math.cos(math.pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    #print('new lr: %.10f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
