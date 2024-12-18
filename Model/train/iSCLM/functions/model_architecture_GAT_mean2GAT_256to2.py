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
# from torch_geometric.utils import dense_to_sparse, dropout_adj, true_positive, true_negative, false_positive, \
#     false_negative, precision, recall, f1_score
from torch_geometric.nn import knn_graph, dense_diff_pool, global_add_pool, global_mean_pool, DataParallel,global_max_pool
from torch_geometric.nn import JumpingKnowledge as jp
from torch_geometric.utils import to_dense_batch

from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn import metrics


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.nhid = 256
        self.feature = 64

        self.conv1 = GATConv(self.feature, 64, 2)
        self.bn1 = torch.nn.BatchNorm1d(64 * 2)
        self.conv2 = GATConv(64 * 2, 32, 2)
        self.bn2 = torch.nn.BatchNorm1d(32 * 2)
        self.bn256 = torch.nn.BatchNorm1d(256)

        self.jpn = jp('max')  # 'max'

        # self.fc0 = Linear(self.nhid, 256)
        # self.fc1 = Linear(256, self.nhid // 2)
        self.fc2 = Linear(self.nhid // 2, self.nhid // 4)  # no use
        self.fc3 = Linear(self.nhid // 4, 2)  # no use
        self.fc4 = Linear(64, 2)  # no use
        self.fc5 = Linear(64, 256)
        self.fc256to2 = Linear(256, 4)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        xconv2 = self.bn2(x)

        # select_index = global_sort_pool(x, batch, 20)
        # x = global_max_pool(x, batch)
        # x = global_mean_pool(x, batch)
        xconv2 = global_mean_pool(xconv2, batch)

        # x = F.relu(self.fc0(x))
        # x = self.fc1(x)
        # x = F.relu(F.dropout(x, p=0.5, training=self.training))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        xconv2 = self.fc5(xconv2)
        _xconv2 = F.normalize(xconv2, dim=1)
        xconv2 = F.relu(xconv2)
        xconv2 = self.bn256(xconv2)
        # x = self.fc4(x)
        x = self.fc256to2(xconv2)
        return F.log_softmax(x, dim=1), _xconv2  #


def global_sort_pool(x, batch, k):
    r"""The global pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.
    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        k (int): The number of nodes to hold for each graph.
    :rtype: :class:`Tensor`
    """
    fill_value = x.min().item() - 1
    batch_x, _ = to_dense_batch(x, batch, fill_value)
#     print(batch_x)
    B, N, D = batch_x.size()

    copy_all = []#copy.deepcopy(batch_x).tolist()
    for i in batch_x:
        copy_all.append(i.tolist())
    _ , perm = batch_x[:, :, -1].sort(dim=-1, descending=True)
    arange = torch.arange(B, dtype=torch.long, device=perm.device) * N
    perm = perm + arange.view(-1, 1)

    batch_x = batch_x.view(B * N, D)
    batch_x = batch_x[perm]
    batch_x = batch_x.view(B, N, D)


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

