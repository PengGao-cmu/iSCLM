import os
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import torch_geometric.datasets
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
from torchvision.models.resnet import model_urls
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
from pre_functions.utils_clpca import generate_dataset
import torch
np.random.seed(0)
seed_value = 0  # set random
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
from pre_functions.model_architecture_GAT_mean2GAT_1226 import Net, global_sort_pool


def add_label(dic_name_data, dic_id_label):
    name_neg = []
    name_pos = []
    for key, data in dic_name_data.items():
        if key in dic_id_label.keys():
            label = dic_id_label[key]
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


def test(loader, slide_name_order, dic_slide_patch, model):
    model.eval()
    correct = 0
    loss_all = 0
    epoch_pred = []
    epoch_score = []
    epoch_label = []

    dic_slide_select_node = {}
    for data_list in loader:
        detail = [data.data_name for data in data_list]
        print('----------------------------------------------------------process, detail :', detail, data_list)
        output, select_index_list = model(data_list)

        y = torch.cat([data.y for data in data_list]).to(output.device)
        loss = F.nll_loss(output, y)
        pred = output.max(dim=1)[1]
        correct += pred.eq(y).int().sum().item()
        loss_all += loss.item()/y.size(0)

        sm = torch.nn.Softmax(dim=1)
        score = sm(output)
        for s in score:
            epoch_score.append(s[1].item())
        for p in pred:
            epoch_pred.append(p.item())
        for l in y:
            epoch_label.append(l.item())
        select_index_list = select_index_list.tolist()

        new_select_index_list = []
        for i in select_index_list:
            i = [int(j) for j in i]
            new_select_index_list.append(i)

        # select
        for graph, index_list in enumerate(new_select_index_list, 0):
            slide_name = slide_name_order[detail[graph]]
            node_list = dic_slide_patch[slide_name]

            select_node = [node_list[i] for i in index_list]
            dic_slide_select_node[slide_name] = select_node
    return loss_all, correct / len(loader.dataset), epoch_pred, epoch_score, epoch_label, dic_slide_select_node


def main_pre(modelpath,savecsvname,feature_path,feature_name_path,adj_path,label_path,InLearning):

    modelpath = modelpath
    savecsvname = savecsvname

    feature_path =      feature_path
    feature_name_path = feature_name_path
    adj_path =          adj_path
    label_path, namestr, labelstr = label_path, 'Pfile', 'label'



    dataset = pd.read_csv(label_path)
    df = pd.DataFrame(dataset)
    dic_id_label = {}
    name = df[namestr].values.tolist()
    label = df[labelstr].values.tolist()
    label = [int(i) for i in label]
    for index, n in enumerate(name, 0):
        dic_id_label[n] = label[index]

    dic_name_data, _data_name, slide_name_order, dic_slide_patch = \
        generate_dataset(64, feature_path, feature_name_path, adj_path, use_knn_graph='False', pca_right_now='True')
    name_neg, name_pos, dic_name_data = add_label(dic_name_data, dic_id_label)
    print('**********************************************start pre*********************************************')
    old_test_dataset = []
    old_test_name = []
    test_name = []
    test_dataset = []
    index_pos = np.array([i for i in range(len(name_pos))])
    index_neg = np.array([i for i in range(len(name_neg))])
    for i in index_pos:
        name = name_pos[i]
        old_test_dataset.append(dic_name_data[name])
        old_test_name.append(name)
    for i in index_neg:
        name = name_neg[i]
        old_test_dataset.append(dic_name_data[name])
        old_test_name.append(name)
    indexlist = [i for i in range(len(old_test_name))]
    random.shuffle(indexlist)
    for i in indexlist:
        test_dataset.append(old_test_dataset[i])
        test_name.append(old_test_name[i])

    print('total of test{}...'.format(len(test_dataset)))
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    test_loader = DataListLoader(test_dataset, batch_size=1, shuffle=False)

    # ------------------------------------------------------TEST--------------------------------------------------------
    nameall = list(name_pos) + list(name_neg)
    dic_multi_model_pred = {}
    for i in nameall:
        dic_multi_model_pred[i] = []
    dic_multi_model_score = {}
    for i in nameall:
        dic_multi_model_score[i] = []
    dic_multi_model_attn = {}
    for i in nameall:
        dic_multi_model_attn[i] = []
    # Load Model for Inference
    model = Net(InLearning)
    for layer in model.children():
        layer.reset_parameters()

    for i,j in model.named_parameters():
        print(i,j.shape)
    a = torch.load(modelpath)

    model.load_state_dict(a, strict=False)
    model.eval()
    model = DataParallel(model)
    # model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_loss, test_acc, pred_label, pred_score, epoch_label, epoch_slide_select_patch = \
        test(test_loader, slide_name_order,dic_slide_patch,model,)

    y = epoch_label
    y_score = pred_score
    test_auc2 = metrics.roc_auc_score(y, y_score)
    print('**********************************************Finish *********************************************')
    print('test acc auc :', test_acc, test_auc2)
    # print('pred_label, pred_score, epoch_label', pred_label, pred_score, epoch_label)
    for names, labels in zip(test_name, pred_label):
        dic_multi_model_pred[names].append(labels)
    for names, scores in zip(test_name, pred_score):
        dic_multi_model_score[names].append(scores)
    for names in test_name:
        if names not in epoch_slide_select_patch.keys():
            print(names)
            continue
        else:
            dic_multi_model_attn[names].append(list(epoch_slide_select_patch[names]))
    print('saving result to ......', savecsvname)
    nameList = []
    prelabelList = []
    for i, j in dic_multi_model_pred.items():
        nameList.append(i)
        prelabelList.append(j[0])

    res = pd.DataFrame()
    res['name'] = nameList
    res['pre_label'] = pred_label
    res['score'] = pred_score
    res['true_label'] = epoch_label
    os.mkdir(fr'./{savecsvname}_predict')
    res.to_csv(fr'./{savecsvname}_predict/predlabel_{savecsvname}.csv',index=False)
    with open(fr'./{savecsvname}_predict/predlabel_{savecsvname}.json','w') as outfile:
        json.dump(dic_multi_model_pred, outfile)
    with open(fr'./{savecsvname}_predict/predscore_{savecsvname}.json','w') as outfile:
        json.dump(dic_multi_model_score, outfile)
    with open(fr'./{savecsvname}_predict/patchrank_{savecsvname}.json','w') as outfile:
        json.dump(dic_multi_model_attn, outfile)


if __name__ == '__main__':
    modelpath = 'bl_duibixuexi_nozengliang_random0_3_notsoftmax__model_parameter_46e-4_blct_blleinei_1226add'
    savecsvname = 'duibixuexi_bl_lr_img_oriloss_qianzhan'
    feature_path = '/data1/gnn/Test/qianzhan_fea_pca64.json'
    feature_name_path = '/data1/gnn/Test/qianzhan_name.json'
    adj_path = '/data1/gnn/Test/qianzhan_adj.json'
    label_path = r'/data1/gnn/Test/qianzhan_del_less64.csv'

    InLearning = False # If the model being used is an incremental model, select True; if not, select False.
    main_pre(modelpath,savecsvname,feature_path,feature_name_path,adj_path,label_path,InLearning)