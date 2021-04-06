import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

from utils import mlp

def make_policy_for_env(env, **kwargs):
    pi_hid = kwargs['pi_hid']
    pi_l = kwargs['pi_l']
    if isinstance(env.action_space, Box):
        pi = GaussianPolicy(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_sizes=[pi_hid] * pi_l,
        )
    elif isinstance(env.action_space, Discrete):
        pi = CategoricalPolicy(
            obs_dim=env.observation_space.shape[0],
            act_n=env.action_space.n,
            hidden_sizes=[pi_hid] * pi_l,
        )
    else:
        raise RuntimeError("Bad Env!")
    return pi

class GaussianPolicy(nn.Module):
    """
    Gaussian Policy Network (Actor) for continuous control
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + hidden_sizes + [act_dim], nn.Tanh)

    def forward(self, obs, requires_grad=True):
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        if not requires_grad:
            with torch.no_grad():
                mu = self.mu_net(obs)
                std = torch.exp(self.log_std)
        else:
            mu = self.mu_net(obs)
            std = torch.exp(self.log_std)

        pi = Normal(mu, std)
        a = pi.sample()
        return a.numpy(), pi.log_prob(a).sum(axis=-1)  # 要把动作每个维度的 log prob 求和（独立假设）


class CategoricalPolicy(nn.Module):
    """
    Categorical Policy Network (Actor) for discrete control
    """

    def __init__(self, obs_dim, act_n, hidden_sizes):
        super().__init__()
        self.logits_net = mlp([obs_dim] + hidden_sizes + [act_n], nn.Tanh)

    def forward(self, obs, requires_grad=True):
        """
        对于当前状态 obs，输出动作 a 以及 log(pi(a|obs))，其中 logprob 包含梯度信息
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        if not requires_grad:
            with torch.no_grad():
                logits = self.logits_net(obs)
        else:
            logits = self.logits_net(obs)

        pi = Categorical(logits=logits)
        a = pi.sample()
        return a.numpy(), pi.log_prob(a)
