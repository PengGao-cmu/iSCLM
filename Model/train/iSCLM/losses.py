from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        print('loss 超参数',temperature,base_temperature)
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.print_debug = False

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        if self.print_debug:
            print('start loss')

        __anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature) # anchor_feature = contrast_feature
        _anchor_feature0 = anchor_feature[0,:]
        _anchor_feature1 = anchor_feature[1, :]
        # for numerical stability 减去每行最大值 提高数值稳定性
        __logits_max, _ = torch.max(__anchor_dot_contrast, dim=1, keepdim=True)
        logits = __anchor_dot_contrast - __logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        __logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
        mask = mask * __logits_mask
        exp_logits = torch.exp(logits) * __logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        moremask = []  # 仅计算病理和影像 / 病理和影像之间特征+影像类内 / 病理和影像之间特征+病理内类内 / 所有
        for i in range(batch_size):
            #moremask.append([0] * batch_size + [1] * batch_size)
            moremask.append([1] * batch_size + [1] * batch_size)
        for i in range(batch_size):
            #moremask.append([1] * batch_size + [0] * batch_size)
            moremask.append([1] * batch_size + [1] * batch_size)
        moremask = torch.from_numpy(np.array(moremask)).to(device)
        #print(moremask)
        mask = mask * moremask

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


if __name__ == '__main__':
    loss = SupConLoss()
    img = torch.tensor([[1,9,3,4,10,6],[1,2,3,4,33,1],[5,6,5,6,5,7],[1,8,8,9,9,7]])
    patho = torch.tensor([[24,9,10,14,16,2],[24,19,10,14,16,2],[24,5,3,2,5,4],[21,2,2,2,3,4]])
    # print(img.shape)
    # print(bl.shape)
    labels = torch.tensor([1,1,0,0])
    features = torch.stack([img,patho], dim=1)
    a = loss.forward(features, labels)
    # https://github.com/HobbitLong/SupContrast/blob/master/losses.py