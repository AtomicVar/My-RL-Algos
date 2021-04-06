import torch
import numpy as np
import scipy.signal
from torch import nn


def count_params(module):
    """
    统计一个神经网络中参数的总数
    """
    return sum([np.prod(p.shape) for p in module.parameters()])


def normalize(t):
    """
    正则化一个 tensor，减去均值，除以标准差
    """
    m = torch.mean(t)
    std = torch.std(t)
    return (t - m) / std


def discounted_sum(x, discount, ret_tensor=False):
    """
    计算 Reward-to-Go 向量，来自 rllab 的神奇代码

    输入：
        vector x, 
        [x0, 
         x1, 
         x2]
    输出：
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    ret = scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return torch.as_tensor(ret.copy()) if ret_tensor else ret


def mlp(sizes, activation, output_activation=nn.Identity):
    """
    用于构建多层神经网络（MLP）
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

