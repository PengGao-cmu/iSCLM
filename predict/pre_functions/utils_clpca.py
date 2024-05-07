import os
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
from torch_geometric.data import DataListLoader
from torch_geometric.nn import SAGEConv, GINConv, DenseGCNConv, GCNConv, GATConv, EdgeConv
from torch_geometric.utils import dense_to_sparse, dropout_adj, true_positive, true_negative, false_positive, \
    false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool, DataParallel
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics


def generate_dataset(pca_feature_dimension, feature_path, feature_name_path, slide_patch_adj_path, use_knn_graph,
                     pca_right_now):
    # load slide:feature
    path = feature_path
    with open(path) as f:
        dic_slide_feature = json.load(f)

    # load slide:[patch]
    path = feature_name_path
    with open(path) as f:
        dic_slide_patch = json.load(f)

    path = slide_patch_adj_path
    with open(path) as f:
        dic_slide_adj_info = json.load(f)

    data_list = []
    data_name = []
    dic_name_data = {}

    feature = {}
    slide_name_order = []
    count = 0

    for key, patch in dic_slide_patch.items():
        if key not in dic_slide_feature.keys():
            print(key,'not in')
            continue

        imagelist = patch
        N = len(imagelist)
        x = dic_slide_feature[key]
        print('load data :',key, np.array(x).shape,)
        print(len(imagelist),imagelist[:10],)
        # print('if len(x) < pca_feature_dimension:', len(x))
        # if len(x) < pca_feature_dimension:
        #     print('if len(x) < pca_feature_dimension:', len(x))
        #     continue
        # if pca_right_now == 'True':
        #     x = np.array(x)
        #     x = PCA(n_components=pca_feature_dimension, random_state=0).fit_transform(x)

        x = np.array(x)
        x = torch.Tensor(x)


        if use_knn_graph == 'True':
            patch_name_list = dic_slide_patch[key]
            coordinates = torch.Tensor([get_coordinate(i) for i in patch_name_list])
            batch = torch.LongTensor(np.zeros(len(patch_name_list)))
            edge_index = knn_graph(coordinates, k=500, batch=batch, loop=True)
        else:
            adj_info = dic_slide_adj_info[key]
            adj = generate_adj(imagelist, N, adj_info)
            adj = torch.Tensor(adj)
            print('adj',np.array(adj).shape,adj)
            edge_index, edge_attr = dense_to_sparse(adj)
        slide_name_order.append(key)

        print(x.shape, x)
        print(edge_attr.shape,edge_attr,np.unique(edge_attr))
        print(edge_index.shape,edge_index)
        data = td.Data(edge_index=edge_index, x=x)

        data.data_name = torch.tensor([count], dtype=torch.int)
        count += 1
        print('count: {}, slide: {}'.format(count, key))
        data_list.append(data)
        data_name.append(key)
        dic_name_data[key] = data
        # if count ==20:
        #     break
    return dic_name_data, data_name, slide_name_order, dic_slide_patch


def generate_adj(patch_list, N, adj_info):
    adj = np.zeros((N, N))
    for index_r, patch_r in enumerate(patch_list, 0):
        neighbors = adj_info[patch_r]
        for index_c, patch_c in enumerate(patch_list, 0):
            if index_r == index_c:
                adj[index_r][index_c] = 1
            if patch_c in neighbors:
                adj[index_r][index_c] = 1
    return adj


def get_coordinate(name):
    sp = name.split('.')[0]
    sp = sp.split('_')
    x = int(sp[1])
    y = int(sp[2])
    return x, y





