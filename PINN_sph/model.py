#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PINN_sph 
@File    ：model.py
@Author  ：LiangL. Yan
@Date    ：2023/1/09 20:09
"""

import torch
import torch.nn as nn
import torch.optim as optim

# 引力常数 - m^3/(kg*s^2)
G = 6.67408e-11


# util
####################
def print_network(model, name):
    """Print out the information of network."""
    nums_params = 0
    for p in model.parameters():
        nums_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(nums_params))


def print_net_params(model):
    """Print out the information of network params."""
    for p in model.parameters():
        print(p)
####################


class FCNet(nn.Module):
    """FC Neural Network."""
    def __init__(self, layers, w_init=True, active=nn.Tanh()):
        super(FCNet, self).__init__()

        # Parameters
        self.depth = len(layers) - 1
        self.active = active

        # Layers list
        layer_list = list()
        for layer in range(self.depth - 1):
            layer_list.append(
                nn.Linear(layers[layer], layers[layer+1])
            )
            layer_list.append(active)
        layer_list.append(nn.Linear(layers[-2], layers[-1]))

        # Net
        self.main = nn.Sequential(*layer_list)

        # Initialize parameters
        if w_init:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.main(x)


class PINNs(nn.Module):
    """The Basic PINNs(Physics Informed Neural Network) model."""
    def __init__(self, basic_net, optimizer=optim.Adam):
        super(PINNs, self).__init__()

        # Neural Network
        self.net = basic_net

        # optimizer
        self.optimizer = optimizer

    def forward(self, x):
        return self.net(x)

    def equation(self, **kwargs):
        """PDE."""
        pass


class GravityConv(nn.Module):
    def __init__(self):
        super(GravityConv, self).__init__()

    def forward(self):
        pass


class GravityNet(nn.Module):
    """The Network to solve gravity."""
    def __init__(self, input_dim, mass):
        super(GravityNet, self).__init__()

        self.sub_net = FCNet([input_dim, input_dim // 2, input_dim // 4,
                              input_dim // 8,
                              input_dim // 4, input_dim // 2,
                              1
                              ])

        self.mass = mass

    def forward(self, X):
        f = G * self.mass * self.sub_net(X.T)
        return f


if __name__ == '__main__':

    net = FCNet([1] + [20]*3 + [1])
    print_network(net, 'FCNet')
