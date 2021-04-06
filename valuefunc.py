import numpy as np
import torch
from torch import nn

from utils import mlp


class VNet(nn.Module):
    """
    Value Network (Critic)
    """

    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.v_net = mlp([obs_dim] + hidden_sizes + [1], nn.Tanh)

    def forward(self, obs):
        """
        输入状态 obs，输出 (v_tensor, v_float)，即第二个不带梯度信息，纯浮点数，用于其他地方，如构造 Advantage 函数
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        v = self.v_net(obs)
        return v, v.detach().numpy().item()
