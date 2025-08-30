from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes: Sequence[int], activation=nn.ReLU, out_act=None) -> nn.Module:
    """
    构建一个多层感知机(MLP)神经网络。

    参数:
        sizes: 整数序列，表示每层的神经元数量。
        activation: 隐藏层使用的激活函数，默认为nn.ReLU。
        out_act: 输出层使用的激活函数，如果为None则不使用激活函数。

    返回:
        nn.Module: 一个包含MLP各层的nn.Sequential对象。
    """
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        """
        Actor网络的构造函数。

        参数:
            s_dim: 状态维度。
            a_dim: 动作维度。
            hidden_sizes: 隐藏层神经元数量的序列，缺省值为(256, 256)。
        """
        super().__init__()
        self.net = mlp([s_dim, *hidden_sizes, a_dim], activation=nn.ReLU, out_act=nn.Tanh)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Critic(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, hidden_sizes: Sequence[int] = (256, 256)) -> None:
        """
        Critic网络的构造函数。

        参数:
            s_dim: 状态维度。
            a_dim: 动作维度。
            hidden_sizes: 隐藏层神经元数量的序列，缺省值为(256, 256)。
        """
        super().__init__()
        self.net = mlp([s_dim + a_dim, *hidden_sizes, 1], activation=nn.ReLU, out_act=None)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算Q(s, a)。

        参数:
            s: 状态，形状为 (batch_size, s_dim)。
            a: 动作，形状为 (batch_size, a_dim)。

        返回:
            q: 价值函数的输出，形状为 (batch_size,)。
        """
        x = torch.cat([s, a], dim=-1)
        q = self.net(x)
        return q.squeeze(-1)

