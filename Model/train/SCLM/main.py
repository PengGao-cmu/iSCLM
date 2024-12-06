import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from PIL import Image
import time
import copy
import json
from math import ceil, sqrt
import random
import pandas as pd
import numpy as np
from random import randint
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.data as td
import torch_geometric.transforms as T
from torch_geometric.data import DataListLoader, DataLoader
from torch_geometric.nn import SAGEConv, GINConv, DenseGCNConv, GCNConv, GATConv, EdgeConv
from torch_geometric.utils import dense_to_sparse, dropout_adj,true_positive, true_negative, false_positive, false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool,DataParallel
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics
# Customized
# from functions.model_architecture_GSA_maxpool import Net, global_sort_pool
from sklearn.model_selection import StratifiedKFold

# from functions.model_architecture_GAT_maxpool import Net, global_sort_pool
from functions.utils_clpca import generate_dataset, oversampling, majority_vote
# img encoder
from datasetutils import adjust_learning_rate_warm_cos,ImgDataset # set_loader
from losses import SupConLoss
from resnet import SupConResNet # ,SupConResNet_load
np.random.seed(0)
seed_value = 0  
random.seed(seed_value)
torch.manual_seed(seed_value)   
torch.cuda.manual_seed(seed_value)      
torch.cuda.manual_seed_all(seed_value)  
from functions.model_architecture_GAT_mean2GAT_256to2 import Net, global_sort_pool


#############################################    Train and Test Function   #############################################
def add_label(dic_name_data, dic_id_label):
    name_neg = []
    name_pos = []
    for key, data in dic_name_data.items():
        id = key
        if id in dic_id_label.keys():
            label = dic_id_label[id]
            flag = label
            y = torch.from_numpy(np.array([label]))
            if flag == 1:
                data.y = y
                name_pos.append(key)
                dic_name_data[key] = data
            else:
                data.y = y
                name_neg.append(key)
                dic_name_data[key] = data
    return name_neg, name_pos, dic_name_data


def loadtest(csvpath,dic_name_data): 
    testset = pd.read_csv(csvpath)
    testset = pd.DataFrame(testset)
    testnamelist = testset['pathologyfile']
    old_test_dataset = []
    old_test_name = []
    test_name = []
    test_dataset = []
    for name in testnamelist:
        if name in dic_name_data.keys():
            old_test_dataset.append(dic_name_data[name])
            old_test_name.append(name)
    test_dataset = old_test_dataset
    test_name = old_test_name
    return test_dataset, test_name


def test(loader, slide_name_order, dic_slide_patch):
    celoss = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    correct = 0
    loss_all = 0
    epoch_pred = []
    epoch_score = []
    epoch_label = []
    dic_slide_select_node = {}
    blloader, imgloader, imgloader2 = loader
    for data_list, imglist, imglist2 in zip(blloader, imgloader, imgloader2):
        imglist = imglist[0].to(device)
        imglist2 = imglist2[0].to(device)
        _, output = image_encoder((imglist,imglist2))
        y = data_list.y.to(output.device)
        loss = celoss(output, y.long())
        pred = output.max(dim=1)[1]
        correct += pred.eq(y).int().sum().item()
        loss_all += loss.item() / y.size(0)
        sm = torch.nn.Softmax()
        score = sm(output)

        for s in score:
            epoch_score.append(s[1].item())
        for p in pred:
            epoch_pred.append(p.item())
        for l in y:
            epoch_label.append(l.item())
        dic_slide_select_node = []

    return loss_all, correct / len(blloader.dataset), epoch_pred, epoch_score, epoch_label, dic_slide_select_node

#################################################### START! ############################################################
print('start ...............')
init_lr = 1e-4 # 3
num_epochs = 200  # 50 # 60
# 生成name label字典 name顺序为表格顺序
dataset = pd.read_csv(r'../table/demo_trainANDvalid.csv', )
df = pd.DataFrame(dataset)
dic_id_label = {}
name = df['pathologyfile'].values.tolist()
label = df['label'].values.tolist()
label = [int(i) for i in label]
false_label_slide = []
for index, n in enumerate(name, 0):
    dic_id_label[n] = label[index]  # id or name 是key label是value
# ------------------------TRAIN TEST JSON--------------------------#
feature_path5 = r'../demo_pathology/240626-3dataset_fea_pca64.json'
feature_name_path5 = r'../demo_pathology/240626-3dataset_name.json'
adj_path5 = r'../demo_pathology/240626-3dataset_adj.json'
dic_name_data, _, slide_name_order, dic_slide_patch = generate_dataset(64, feature_path5, feature_name_path5, adj_path5,
                                                                       'False', 'True')  # use_knn_graph, pca_right_now)
name_neg, name_pos, dic_name_data = add_label(dic_name_data, dic_id_label)
print('数据量， name', len(dic_name_data.keys()))
# CTdata
label_CTS = dataset['ctnumsame_filename']
label_file = dataset['pathologyfile']
dic_file_ct = {}
for i,j in zip(label_file, label_CTS):
    dic_file_ct[i]=j

# GCN
TRAIN_LOSS = []
TEST_LOSS = []
TRAIN_AUC = []
TEST_AUC = []
TRAIN_ACC = []
TEST_ACC = []

trainvalid = pd.read_csv(r'../table/demo_neoadjuvant_trainANDvalid.csv')

savepath = 'SCL_softmax_tmp0.1_shuffle_right-tmp0.1_BLCT_ORI_loss_img_lr_bl-6-17-ORIloss'
if not os.path.exists(fr'./{savepath}'):
    os.mkdir(fr'./{savepath}')

if __name__ == '__main__':

    nnn = 1
    fold_random = nnn
    fold = StratifiedKFold(n_splits=3, random_state=nnn, shuffle=True) #In the actual training process, 'n_splits=10' is used. However, for the purpose of this demo, we use 'n_splits=3' as an example.
    Xx = trainvalid['pathologyfile']
    Yy = np.array([int(i) for i in trainvalid['label']])
    randnum = 0
    for a, b in fold.split(X=Xx, y=Yy):
        randnum +=1
        trainx, trainy = list(Xx[a]), list(Yy[a])
        validx, validy = Xx[b], Yy[b]

        print('len(trainy),len(validy):', len(trainy), len(validy))
        trainpath, validpath = fr'./{savepath}/1118GAT{nnn}{randnum}tmp_train_load.csv', \
                               fr'./{savepath}/1118GAT{nnn}{randnum}tmp_valid_load.csv'
        pd.DataFrame(np.array([list(trainx), trainy]).T, columns=['pathologyfile', 'label']).to_csv(trainpath,index=False)
        pd.DataFrame(np.array([list(validx), validy]).T, columns=['pathologyfile', 'label']).to_csv(validpath,index=False)
        # TRAIN------------------------------------------------------
        train_dataset, trainname = loadtest(trainpath, dic_name_data)
        indexlist = [i for i in range(len(trainname))]
        random.shuffle(indexlist)
        _train_dataset = []
        _trainname = []
        for i in indexlist:
            _train_dataset.append(train_dataset[i])
            _trainname.append(trainname[i])
        train_loader = DataLoader(_train_dataset, batch_size=72, shuffle=False)  # , num_worker=32)
        # train pathology data and CT data
        traindata0 = ImgDataset(r'../demo__CT_process_trainvalid',[dic_file_ct[i] for i in _trainname],test=False)
        traindata1 = ImgDataset(r'../demo__CT_process_trainvalid',[dic_file_ct[i] for i in _trainname],niiname='lymph_resam.nii.gz',test=False)
        CT_train_loader0 = torch.utils.data.DataLoader(traindata0, num_workers=1, batch_size=72, shuffle=False, )
        CT_train_loader1 = torch.utils.data.DataLoader(traindata1, num_workers=1, batch_size=72, shuffle=False, )

        # traindata for test pathology data
        train_loader_test = DataLoader(train_dataset, batch_size=8, shuffle=False)
        # valid pathology data
        valid_dataset, validname = loadtest(validpath, dic_name_data)
        test_loadervalid = DataLoader(valid_dataset, batch_size=8, shuffle=False,)
      

        # CT data
        traindata0 = ImgDataset(r'../demo__CT_process_trainvalid',[dic_file_ct[i] for i in trainname])
        traindata1 = ImgDataset(r'../demo__CT_process_trainvalid',[dic_file_ct[i] for i in trainname],niiname='lymph_resam.nii.gz')
        CT_train_loader_test = torch.utils.data.DataLoader(traindata0, num_workers=1, batch_size=8, shuffle=False, )
        CT_train_loader_test2 = torch.utils.data.DataLoader(traindata1, num_workers=1, batch_size=8, shuffle=False, )

        traindata0 = ImgDataset(r'../demo__CT_process_trainvalid',[dic_file_ct[i] for i in validname])
        traindata1 = ImgDataset(r'../demo__CT_process_trainvalid',[dic_file_ct[i] for i in validname],niiname='lymph_resam.nii.gz')
        CT_valid = torch.utils.data.DataLoader(traindata0, num_workers=1, batch_size=8, shuffle=False, )
        CT_valid2 = torch.utils.data.DataLoader(traindata1, num_workers=1, batch_size=8, shuffle=False, )

        print('# of train: {} | # of valid: {}'.format(len(train_dataset), len(valid_dataset)))
        print("-------------------Let's use-------------------", torch.cuda.device_count(), "GPUs!")
        model = Net()  # .to(device)
        for layer in model.children():
            layer.reset_parameters()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #model = DataParallel(model)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay = 1e-2)
        scl_loss = SupConLoss() # scl_loss = scl_loss.cuda()
        ce_loss = nn.CrossEntropyLoss()
        image_encoder = SupConResNet()
        image_encoder.to(device)
        img_optimizer1 = torch.optim.AdamW(image_encoder.parameters(), lr=0.0001)

        for epoch in range(1,num_epochs+1):
            print('train................')
            model.train()
            image_encoder.train()
            loss_all = 0
            correct = 0
            epoch_label = []
            epoch_score = []
            train_pred = []
            adjust_learning_rate_warm_cos(optimizer, epoch, num_epochs, lr_max=init_lr)
            for data_list, img_list,  img_list1 in zip(train_loader, CT_train_loader0,CT_train_loader1):
                optimizer.zero_grad()
                img_optimizer1.zero_grad()
                data_list = data_list.to(device)  
                img_list = img_list[0].to(device)
                img_list1 = img_list1[0].to(device)

                _result_, output = model(data_list)
                img_features, result = image_encoder((img_list, img_list1))

                # 联合LOSS
                y = data_list.y.to(output.device)
                allfeatures = torch.stack([img_features,output], dim=1)
                scl_l = scl_loss(allfeatures, y.long())
                #nll = torch.nn.NLLLoss()
                ce_l = ce_loss(result, y.long())
                loss = 0.5*scl_l + 0.5*ce_l
              
                # 参数回传
                loss.backward()
                optimizer.step()
                img_optimizer1.step()

                pred = result.max(dim=1)[1]
                loss_all += loss.item()/y.size(0)
                correct += pred.eq(y).int().sum().item()

                sm = torch.nn.Softmax()
                score = sm(result) # score = result
                for p in pred:
                    train_pred.append(p.item())
                for s in score:
                    epoch_score.append(s[1].item())
                for l in y:
                    epoch_label.append(l.item())
            train_loss, train_acc, train_pred, train_score, train_label = loss_all, correct / len(train_loader.dataset), train_pred, epoch_score, epoch_label

            print('valid..................')
            _loss, test_acctr, pred_labeltr, pred_scoretr, epoch_labeltr, _ = \
                test((train_loader_test,CT_train_loader_test,CT_train_loader_test2), slide_name_order,dic_slide_patch, )
            test_lossVa, test_accVa, pred_labelVa, pred_scoreVa, epoch_labelVa, _ = \
                test((test_loadervalid,CT_valid,CT_valid2,),slide_name_order,dic_slide_patch, )
         


            # evaluate model
            test_auctr = metrics.roc_auc_score(epoch_labeltr, pred_scoretr)
            test_aucVa = metrics.roc_auc_score(epoch_labelVa, pred_scoreVa)
            _tmplabel, _tmpscore = list(epoch_labelVa), list(pred_scoreVa)
            test_allVa = metrics.roc_auc_score(_tmplabel, _tmpscore)
            TMPRESULT = []
            result_acc_auc = []
            print('Epoch: {:03d} :'.format(epoch),rf'random{fold_random}_{randnum}')

            print(
                'Train Loss: {:.7f} Train acc: {:.7f} train auc {:.7f} |Valid Loss: {:.7f} Valid Acc : {} Valid Auc: {:.7f}'.
                format(train_loss, test_acctr, test_auctr, test_lossVa, test_accVa, test_aucVa))
            result_acc_auc.append([test_acctr, test_accVa,
                                   test_auctr, test_aucVa])  
            if test_aucVa > 0.80:
                print('best_test_auc')
                result_acc_auccsv = pd.DataFrame(result_acc_auc)
                result_acc_auccsv.to_csv(fr'./{savepath}/ACCAUC_epoch_random{fold_random}_{randnum}__ep{epoch}.csv')
                #modelsave = fr"./{savepath}/random{fold_random}_{randnum}_notsoftmax__model_parameter_{epoch}e-4.pkl"
                #print('save : ', modelsave)
                #torch.save(model.state_dict(), modelsave)
                modelsave = fr"./{savepath}/random{fold_random}_{randnum}_notsoftmax__model_parameter_{epoch}e-4_imgmodel.pkl"
                print('save : ', modelsave)
                torch.save(image_encoder.state_dict(), modelsave)

                pddata = pd.DataFrame()
                pddata['name'] = trainname + validname
                pddata['prelabel'] = pred_labeltr + pred_labelVa
                pddata['score'] = pred_scoretr + pred_scoreVa
                pddata['truelabel'] = epoch_labeltr + epoch_labelVa
                pddata['data'] = ['train'] * len(trainname) + ['valid'] * len(validname)
                pddata.to_csv(fr'./{savepath}/train_score_epoch_random{fold_random}_{randnum}__ep{epoch}.csv')

     