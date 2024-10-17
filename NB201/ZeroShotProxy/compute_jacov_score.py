'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

-----------
Note from the authors of AZ-NAS

The code is modified from the implementation of ZenNAS [https://github.com/idstcv/ZenNAS/blob/d1d617e0352733d39890fb64ea758f9c85b28c1a/ZeroShotProxy/compute_zen_score.py]

We revise the code as follows:
1. Make it compatible with NAS-Bench-201
'''



import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def init_model(model, method='kaiming_norm_fanin'):
    if method == 'kaiming_norm_fanin':
        model.apply(kaiming_normal_fanin_init)
    else:
        raise NotImplementedError
    return model

def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)

    y = net(x)

    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eigh(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1.0 / (v + k))

def compute_jacov_score(model, gpu, trainloader, resolution, batch_size, mixup_gamma=1e-2, repeat=1, fp16=False):
    model.train()
    model.cuda()
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    init_model(model, 'kaiming_norm_fanin')

    inputs, targets = next(iter(trainloader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    try:
        # Compute gradients (but don't apply them)
        jacobs, labels = get_batch_jacobian(model, inputs, targets)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        jc = eval_score(jacobs, labels)
    except Exception as e:
        print(e)
        jc = np.nan

    info['jacov'] = info

    return jc