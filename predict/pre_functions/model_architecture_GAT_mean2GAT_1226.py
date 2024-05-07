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
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool, DataParallel,global_max_pool
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics


class Net(nn.Module):
    def __init__(self, InLearning):
        super(Net, self).__init__()
        self.nhid = 256
        self.feature = 64 # 64

        self.conv1 = GATConv(self.feature, 64, 2)
        self.bn1 = torch.nn.BatchNorm1d(64 * 2)
        self.conv2 = GATConv(64 * 2, 32, 2)
        self.bn2 = torch.nn.BatchNorm1d(32 * 2)
        self.bn256 = torch.nn.BatchNorm1d(256)

        self.fc5 = Linear(64, 256)
        if InLearning:
            self.fc256to2 = Linear(256, 4)
        else:
            self.fc256to2 = Linear(256, 2)

    def forward(self, data):
        #h = []
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        #h.append(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        xconv2 = self.bn2(x)
        #h.append(x)

        select_index = global_sort_pool(xconv2, batch,)
        xconv2 = global_mean_pool(xconv2,batch)

        xconv2 = self.fc5(xconv2)
        _xconv2 = F.normalize(xconv2, dim=1)
        xconv2 = F.relu(xconv2)
        xconv2 = self.bn256(xconv2)
        x = self.fc256to2(xconv2)
        return F.log_softmax(x[:,-2:],dim=1), select_index


def global_sort_pool(x, batch,):
    fill_value = x.min().item() - 1
    batch_x, _ = to_dense_batch(x, batch, fill_value)

    B, N, D = batch_x.size()
    copy_all = []
    for i in batch_x:
        copy_all.append(i.tolist())

    _, perm = batch_x[:, :, -1].sort(dim=-1, descending=True)

    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)
    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)

    k = N
    if N >= k:
        batch_x = batch_x[:, :k].contiguous()
        copy_select = batch_x.tolist()
        select_index = []
        for ori_graph, k_graph in zip(copy_all, copy_select):
            node_index = []
            for node in k_graph:
                node_index.append(ori_graph.index(node))
            select_index.append(node_index)
    else:
        expand_batch_x = batch_x.new_full((B, k - N, D), fill_value)
        batch_x = torch.cat([batch_x, expand_batch_x], dim=1)
    return torch.Tensor(select_index).cuda()

