import os
import PIL.ImageShow
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as trans
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from torch.optim import lr_scheduler
import torchvision.models as models
import os
import SimpleITK as sitk
import cv2
import matplotlib.pyplot as plt
import sklearn
import random
import sklearn.metrics as mr
from sklearn.model_selection import StratifiedKFold
import matplotlib.colors as mcolors

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



# setting a random seed
setup_seed(180)


class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, rootpath, csvpath, niiname='weiai_resam.nii.gz', test=False):
        self.csv = pd.read_csv(csvpath,  encoding='ANSI')
        self.names = [str(i) for i in self.csv['ctnumsame_filename']]
        self.dic = self.generate_dic()
        self.imgpath = [os.path.join(rootpath, str(i), niiname) for i in self.names]
        self.label = [self.dic[i] for i in self.names]
        self.trans_ct = trans.Compose([
            trans.ToPILImage(),
            trans.Resize((224, 224)),
            trans.RandomHorizontalFlip(),
            trans.ToTensor(),
        ])
        self.test = test
        self.trans_cttest = trans.Compose([
            trans.ToPILImage(),
            trans.Resize((224, 224)),
            trans.ToTensor(),
        ])

    def getnames(self):
        return self.names

    def generate_dic(self):
        dic_ct_label = {}
        for i in range(len(self.names)):
            dic_ct_label[self.names[i]] = self.csv['label'][i]
        return dic_ct_label

    def __getitem__(self, index):
        imgpath = self.imgpath[index]
        img = sitk.ReadImage(imgpath)
        img_3C = sitk.ReadImage(imgpath.replace('.nii', '3ceng.nii'))
        # imgnp_ori= sitk.GetArrayFromImage(img)[0,:,:]
        imgnp_ori = sitk.GetArrayFromImage(img)
        img_3Cnp = sitk.GetArrayFromImage(img_3C)
        if int(np.max(imgnp_ori)) == 0:
            imgnp_ori[imgnp_ori == float(0)] = 255


        imgnp = cv2.threshold(imgnp_ori, 20, 255, cv2.THRESH_BINARY)[1]
        imgnp = np.asarray(imgnp, dtype=np.uint8)
        image_, cnts_, = cv2.findContours(imgnp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(image_[0])

        if len(image_) == 2:
            x2, y2, w2, h2 = cv2.boundingRect(image_[1])
            x, y, w, h = min(x, x2), min(y, y2), max(w2 + x2, w + x) - min(x, x2), max(y2 + h2, y + h) - min(y, y2)

        width = 10
        newimgnp = imgnp_ori[y - width:int(y + h) + width, x - width:int(x + w) + width]
        _imgnp_jiequhou_bz_mask = imgnp[y - width:int(y + h) + width, x - width:int(x + w) + width]
        # plt.imshow(_imgnp_jiequhou_bz_mask)
        # plt.show()

        img_3Cnp_crop = img_3Cnp[:, y - width:int(y + h) + width, x - width:int(x + w) + width]
        if len(image_) >= 3:
            newimgnp = imgnp_ori
            img_3Cnp_crop = img_3Cnp
        # imgnp = np.stack([newimgnp, newimgnp, newimgnp], axis=0)
        imgnp = img_3Cnp_crop
        # print(imgnp.shape)
        # imgnp = np.transpose(imgnp, [1, 2, 0])
        max_, min_ = np.max(imgnp), np.min(imgnp)
        if max_ == min_ == 0:
            imgnp = imgnp
        else:
            imgnp = (imgnp - min_) / (max_ - min_)
        imgnp = imgnp.astype(np.float32)
        imgnp = torch.from_numpy(imgnp)
        if not self.test:
            imgtf = self.trans_ct(imgnp)
        else:
            imgtf = self.trans_cttest(imgnp)
            # imgtf = self.trans_ct(imgnp)
        # print(imgtf.size())  # torch.Size([3, 224, 224])
        # print(type(imgtf))
        # a = np.array(imgtf)
        # plt.imshow(a[1,:,:])
        # plt.show()
        label = self.label[index]
        #from skimage import io, transform
        #_imgnp_jiequhou_bz_mask = transform.resize(_imgnp_jiequhou_bz_mask, (3, 224, 224))
        _imgnp_jiequhou_bz_mask = self.trans_cttest(_imgnp_jiequhou_bz_mask)
        return imgtf, label, _imgnp_jiequhou_bz_mask

    def __len__(self):
        return len(self.imgpath)


class SupConResNet_ZL(nn.Module):
    def __init__(self, ):
        super(SupConResNet_ZL, self).__init__()
        self.res = models.resnet34(pretrained=True)
        self.res2 = models.resnet34(pretrained=True)
        self.fcf = nn.Sequential(nn.Linear(2000, 256),
                                 # nn.BatchNorm1d(512),
                                 nn.ReLU(), nn.BatchNorm1d(256),
                                 # nn.Linear(256, 2),
                                 # nn.Sigmoid())
                                 # active_function
                                 )
        self.fc3 = nn.Linear(256, 4)
        self.act = nn.Softmax()

    def forward(self, _x0, _x1):
        # x0, x1 = x0, x1
        # x0_ = x0 # Variable(x0)

        # x1_ = x1 # Variable(x1)

        x0 = self.res(_x0)

        x1 = self.res2(_x1)

        h = torch.cat([x0, x1], dim=1)

        h = self.fcf(h)


        output = self.fc3(h)
        # output = self.act(output)
        # h = F.normalize(h, dim=1)
        return output[:,-2:]

import shap
def maintrain(modelpath,datapath,datacsv,image_path):
    try:
        validdata_dpxj = ImgDataset(datapath, datacsv, test=True)
        validdata_dpxjload = data.DataLoader(validdata_dpxj, num_workers=0, batch_size=18, shuffle=False, )
        validdata_dpxj2 = ImgDataset(datapath, datacsv, niiname='linba_resam.nii.gz', test=True)
        validdata_dpxjload2 = data.DataLoader(validdata_dpxj2, num_workers=0, batch_size=18, shuffle=False, )
        namedpxj = validdata_dpxj2.getnames()
        # model = NewRes_ZL(0,0)
        model = SupConResNet_ZL()
        print(model)


        a = torch.load(modelpath)
        print(a.keys())
        model.load_state_dict(torch.load(
            modelpath), strict=False)

        for batch, batch2,batchnum in zip(validdata_dpxjload,validdata_dpxjload2,range(20,999,18)):
            print('next load--------------------------')
            images, _, bzmask = batch
            images2, _, lbmask = batch2
            background = [images[:], images2[:]]
            for imagesnum ,testimgnum in zip(range(batchnum-18,batchnum,2),range(2,20,2)):
               for layersnum in range(1,5):

                 test_images = [images[testimgnum-2:testimgnum], images2[testimgnum-2:testimgnum]]
                 Tlabel = ['True label:' + str(np.array(i)) for i in _[testimgnum-2:testimgnum]]
                 filename=str(namedpxj[imagesnum-2])+str(namedpxj[imagesnum-1])
                 layernames=str("layer"+str(layersnum))
                 images_path1= f"\lesion_{layernames}_{filename}.jpg"
                 imagespath=str(image_path+images_path1)
                 test_img = test_images[0].numpy()
                 print(Tlabel)
                 # since we have two inputs we pass a list of inputs to the explainer
                 explainer = shap.GradientExplainer((model, getattr(model.res, "layer"+str(layersnum))[2].conv2), background, local_smoothing=0.2)
                 shap_values = explainer.shap_values(test_images, nsamples=300 )#
                 #print(shap_values)
                 print('-------shap_values')
                 print(np.array(shap_values).shape)  # 2 3 3 224 224 #
                 #print(np.array(shap_values).shape)
                 shap_numpy = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
                 print(np.array(shap_numpy).shape)
                 print('test:', test_img.shape)
                 test_img = np.swapaxes(np.swapaxes(test_img, 1, -1), 1, 2)
                 print(test_img.shape)
                 print(test_img.shape)
                 shap.image_plot(shap_numpy, test_img[:, :, :, 1:2], true_labels=Tlabel, save_path=imagespath)
    except IndexError:
        print("SHAP output complete ")

def get_all_files(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths

if __name__ == '__main__':
    datapath = r"CTpath" #CT Address Storage
    datacsv = r"CTdata.csv" #CT Information File Address
    folder_path = r"modelpath" #Model Address
    file_paths=get_all_files(folder_path)
    image_path=r"imagepath"#Image Output Address
    for modelpath in file_paths:
        print("model path:")
        print(modelpath)
        maintrain(modelpath,datapath,datacsv,image_path)

