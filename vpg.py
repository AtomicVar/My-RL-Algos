import gym
import numpy as np
from numpy.testing._private.utils import requires_memory
import torch
from torch import nn
from torch.distributions.utils import logits_to_probs
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete

from utils import mlp, discounted_sum, count_params, normalize


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy Network (Actor) for continuous control
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        raise NotImplementedError


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

        # 如果不需要梯度，仅用于预测，则在 torch.no_grad() 环境中计算，不知这样是否
        # 会提升性能
        if not requires_grad:
            with torch.no_grad():
                logits = self.logits_net(obs)
        else:
            logits = self.logits_net(obs)

        pi = Categorical(logits=logits)
        a = pi.sample()
        return a.numpy(), pi.log_prob(a)

    def loss(self, logp, adv):
        return -(logp * adv).mean()


class VF(nn.Module):
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
    
    def loss(self, val, ret):
        return ((val - ret) ** 2).mean()


class VPG:
    """
    Vanilla Policy Gradient
    
    (with GAE-Lambda for advantage estimation)
    """

    def __init__(self, env_name, pi_lr, v_lr, pi_hid, pi_l, v_hid, v_l, gamma, lam):
        # 创建环境，并获得环境的状态 shape 以及动作 shape，用于确定神经网络输入输出的 shape
        self.env = gym.make(env_name)

        # 超参数
        self.gamma = gamma
        self.lam = lam

        # 根据环境动作空间是连续还是离散，创建不同类型的策略网络
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Box):
            self.pi = GaussianPolicy(
                obs_dim=obs_dim,
                act_dim=self.env.action_space.shape[0],
                hidden_sizes=[pi_hid] * pi_l,
            )
        elif isinstance(self.env.action_space, Discrete):
            self.pi = CategoricalPolicy(
                obs_dim=obs_dim,
                act_n=self.env.action_space.n,
                hidden_sizes=[pi_hid] * pi_l,
            )
        else:
            raise RuntimeError("Bad Env!")

        # 创建价值网络
        self.v = VF(obs_dim, hidden_sizes=[v_hid] * v_l)

        # 创建优化器
        self.pi_opt = Adam(self.pi.parameters(), pi_lr)
        self.v_opt = Adam(self.v.parameters(), v_lr)

    def train(self, epochs, max_ep_len, log_cycle):
        """
        进行 epochs 轮的训练
        """
        for epoch in range(epochs):
            # 记录样本点
            rew_batch = []  # 用于计算 Rt 以及 At，存放 float
            logp_batch = []  # 带有梯度信息，存放 torch.Tensor
            val_batch = []  # 带有梯度信息的 V(St)，用于优化价值函数以及计算 At，存放 torch.Tensor
            val_batch_float = [] # 不带梯度信息版的 V(St)

            # 与环境交互
            o, ep_ret, ep_len = self.env.reset(), 0, 0
            for t in range(max_ep_len):
                # 环境状态转移
                v, v_float = self.v(o)
                a, logp = self.pi(o)
                o, r, d, _ = self.env.step(a)

                # 存储经验
                logp_batch.append(logp)
                val_batch.append(v)
                val_batch_float.append(v_float)
                rew_batch.append(r)

                ep_ret += r
                ep_len += 1

                if d:
                    break

            # 根据结束状态添加最后一个 reward/val
            if t == max_ep_len:
                # 没有到达 terminal state，需要再运行一步得到下一个状态，因为 delta 的计算需要
                print("Cut out!")
                a, _ = self.pi(o)
                o, r, _, _ = self.env.step(a)
                _, v_float = self.v(o)
                val_batch_float.append(v_float)
                rew_batch.append(r)
                ep_ret += r
            else:
                # 到达 terminal state，则下一个状态的价值直接是 0
                val_batch_float.append(0.0)
                rew_batch.append(0.0)

            # 将 list 转成 np.array 或 torch.tensor，便于后面操作
            val_batch_float = np.stack(val_batch_float)
            rew_batch = np.stack(rew_batch)
            logp_batch = torch.stack(logp_batch)
            val_batch = torch.stack(val_batch)

            # 计算 GAE-Lambda advantage
            deltas = rew_batch[:-1] + self.gamma * val_batch_float[1:] - val_batch_float[:-1]
            adv_batch = discounted_sum(deltas, self.gamma * self.lam, ret_tensor=True)
            # adv_batch = normalize(adv_batch)

            # 计算 Rt (Reward-to-Go)
            ret_batch = discounted_sum(rew_batch, self.gamma, ret_tensor=True)

            # 用样本点对策略网络和价值函数进行一次优化
            self.pi_opt.zero_grad()
            loss_pi = self.pi.loss(logp_batch, adv_batch)
            loss_pi.backward()
            self.pi_opt.step()

            self.v_opt.zero_grad()
            loss_v = self.v.loss(val_batch, ret_batch)
            loss_v.backward()
            self.v_opt.step()

            # 输出
            if epoch % log_cycle == 0:
                print(f"Epoch {epoch}: return = {ep_ret}, length = {ep_len}")

    def simulate(self, rollouts, max_ep_len):
        """
        进行 rollouts 轮的模拟，用于查看训练效果
        """
        total_ret = 0
        total_len = 0
        for rollout in range(rollouts):
            o, ep_ret, ep_len = self.env.reset(), 0, 0
            self.env.render()
            for t in range(max_ep_len):
                a, _ = self.pi(o, requires_grad=False)
                o, r, d, _ = self.env.step(a)
                self.env.render()
                ep_ret += r
                ep_len += 1
                if d:
                    break
            # 打日志
            print(f"Rollout {rollout}: return = {ep_ret}, length = {ep_len}")
            total_ret += ep_ret
            total_len += ep_len
        print(
            f">>> Average return={total_ret / rollouts}, average length = {total_len / rollouts}"
        )


if __name__ == "__main__":
    # 环境
    env_name = "CartPole-v0"

    # 训练参数
    pi_lr = 3e-4
    v_lr = 1e-3
    gamma = 0.99
    lam = 0.97
    max_ep_len = 500
    train_epochs = 1000

    # Actor-Critic 神经网络结构
    pi_hid = 64
    pi_l = 2
    v_hid = 64
    v_l = 2

    # 其他超参
    sim_rollouts = 5
    log_cycle = 100  # 每训练 50 轮进行一次输出

    # 创建 agent
    agent = VPG(env_name, pi_lr, v_lr, pi_hid, pi_l, v_hid, v_l, gamma, lam)

    # 训练前先看看效果
    # agent.simulate(sim_rollouts, max_ep_len)

    # 训练
    agent.train(train_epochs, max_ep_len, log_cycle)

    # 查看仿真效果
    agent.simulate(sim_rollouts, max_ep_len)

