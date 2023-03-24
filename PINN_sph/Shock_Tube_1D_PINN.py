#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PINN_sph 
@File    ：Shock_Tube_1D_PINN.py
@Author  ：LiangL. Yan
@Date    ：2023/3/23 9:22 
"""
##########################################################
## This Code is Modified from Alexandros Papados's code ##
##########################################################

"""
    dU / dt + A * dU / dx = 0   (1)         (x, t) in (0, 1)x(0, 0.2+)
    ==>
    U = [ rho ]         A = [ u, rho, 0 ]
        [ u ]               [ 0, u, 1/rho ]
        [ p ]               [0, gamma*p, u]
"""


# Import libraries
from model import *
from utils import *
import torch
import torch.nn as nn
import numpy as np
import time
import scipy.io
import os
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt


# Seeds
torch.manual_seed(123456)
np.random.seed(123456)

# Create directories if not exist.
log_dir = './work/loss/shock_tube'
model_save_dir = "./model/shock_tube"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Basic Neural Network
class ShockTubeSodPINN(PINNs):
    """The PINN model to solve the shock tube(Sod problem) which is a basic example in SPH."""
    def __init__(self, basic_net, optim_=optim.Adam):
        super(ShockTubeSodPINN, self).__init__(basic_net, optim_)

    def equation(self, X):
        """PDE."""

        gamma = 1.4

        # the network input is x,t; the output is rho, p, u
        y = self.net(X)
        rho = y[:, 0:1]
        p = y[:, 1:2]
        u = y[:, 2:3]

        # 控制方程
        # 1. du / dt + df / dx = 0
        # f1 --- drho_dt + rho * du_dx + u * drho_dx = 0
        dr = gradients(X, rho)
        du = gradients(X, u)

        drdt = dr[:, 0:1]
        drdx = dr[:, 1:2]
        dudx = du[:, 1:2]

        L_f1 = drdt + rho * dudx + u * drdx

        # f2 --- rho * (du_dt + u * du_dx) + dp_dx
        dp = gradients(X, p)

        dpdx = dp[:, 1:2]
        dudt = du[:, 0:1]

        L_f2 = rho * (dudt + u * dudx) + dpdx

        # f3 --- dp_dt + gamma * p * du_dx + u * dp_dx
        dpdt = dp[:, 0:1]

        L_f3 = dpdt + gamma * p * dudx + u * dpdx

        L_f = torch.mean(L_f1**2) + torch.mean(L_f2**2) + torch.mean(L_f3**2)

        return L_f

    def loss_ic(self, x_ic, rho_ic, u_ic, p_ic):
        """Loss of pde init condition."""

        y_ic = self.net(x_ic)
        rho_ic_nn, u_ic_nn, p_ic_nn = y_ic[:, 0], y_ic[:, 2], y_ic[:, 1]

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
                   ((rho_ic_nn - rho_ic) ** 2).mean() + \
                   ((p_ic_nn - p_ic) ** 2).mean()

        return loss_ics


# Initial conditions
def IC(x):
    N = len(x)
    rho_init = np.zeros((x.shape[0]))
    u_init = np.zeros((x.shape[0]))
    p_init = np.zeros((x.shape[0]))

    # rho, p - initial condition
    for i in range(N):
        if (x[i] <= 0.5):
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1

    return rho_init, p_init, u_init


# Solve Euler equations using PINNs
def main():
    # Initialization
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    lr = 0.0005
    num_x = 1000
    num_t = 1000
    num_i_train = 1000
    epochs = 76140
    num_f_train = 11000
    x = np.linspace(-1.5, 3.125, num_x)
    t = np.linspace(0, 0.2, num_t)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    id_ic = np.random.choice(num_x, num_i_train, replace=False)
    id_f = np.random.choice(num_x*num_t, num_f_train, replace=False)

    x_ic = x_grid[id_ic, 0][:, None]
    t_ic = t_grid[id_ic, 0][:, None]
    x_ic_train = np.hstack((t_ic, x_ic))
    rho_ic_train, p_ic_train, u_ic_train = IC(x_ic)

    x_int = X[:, 0][id_f, None]
    t_int = T[:, 0][id_f, None]
    x_int_train = np.hstack((t_int, x_int))
    x_test = np.hstack((T, X))

    # Generate tensors
    x_ic_train = torch.tensor(x_ic_train, dtype=torch.float32).to(device)
    x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

    rho_ic_train = torch.tensor(rho_ic_train, dtype=torch.float32).to(device)
    u_ic_train = torch.tensor(u_ic_train, dtype=torch.float32).to(device)
    p_ic_train = torch.tensor(p_ic_train, dtype=torch.float32).to(device)

    # Initialize neural network
    layers = [2] + [30] * 7 + [3]
    net = FCNet(layers)
    model = ShockTubeSodPINN(net).to(device)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train PINNs
    def train(epoch):
        model.train()

        def closure():
            optimizer.zero_grad()
            loss_pde = model.equation(x_int_train)
            loss_ic = model.loss_ic(x_ic_train, rho_ic_train, u_ic_train, p_ic_train)
            # loss = 0.1*loss_pde + 10*loss_ic
            loss = 0.1 * loss_pde + 10 * loss_ic

            # Print iteration, loss of PDE and ICs
            print(f'epoch {epoch} loss_pde:{loss_pde:.8f}, loss_ic:{loss_ic:.8f}')
            loss.backward()
            return loss

        # Optimize loss function
        loss = optimizer.step(closure)
        loss_value = loss.item() if not isinstance(loss, float) else loss
        # Print total loss
        print(f'epoch {epoch}: loss {loss_value:.6f}')
        print('####################\n')

    # Print CPU
    print('####################')
    print('Start training...')
    print('####################\n')
    tic = time.time()
    for epoch in range(1, epochs+1):
        train(epoch)
    toc = time.time()
    print(f'Total training time: {toc - tic}')

    # current time
    date_time = datetime.datetime.now().strftime("%Y_%m_%d")

    # Save the network parameters
    save_name = 'w-pinn-de_' + str(epochs) + '_' + date_time + '_.ckpt'
    model_save_dir = './model/'
    path = os.path.join(model_save_dir, save_name)
    torch.save(model.net.state_dict(), path)
    print('Saved model checkpoints into {}...'.format(model_save_dir))

    # Evaluate on the whole computational domain
    ##### DATA From SPH
    data_sph = np.loadtxt("./data/shock_tube/SPHBody_WaveBody_0000201640.txt")
    x_sph = data_sph[:, 0]
    t_sph = np.linspace(0.20164, 0.20164, 4000)
    assert x_sph.shape == t_sph.shape

    t_grid_sph, x_grid_sph = np.meshgrid(t_sph, x_sph)
    T_sph = t_grid.flatten()[:, None]
    X_sph = x_grid.flatten()[:, None]

    x_test_sph = torch.tensor(np.hstack((T_sph, X_sph)), dtype=torch.float32).to(device)

    u_pred = to_numpy(model(x_test_sph))

    # save predict result.
    scipy.io.savemat(f'./work/shock_tube/Sod_Shock_Tube_w_pinn_de_{date_time}.mat', {'x': x, 't': t, 'rho': u_pred[:, 0],
                                                              'p': u_pred[:, 1],
                                                              'u': u_pred[:, 2]})


def plot_MAT_FILE(data):
    pass


if __name__ == '__main__':

    main()

    # data = scipy.io.loadmat("./work/shock_tube/Sod_Shock_Tube_w_pinn_de_2023_03_24.mat")
    # solution = np.loadtxt("./data/shock_tube/SPHBody_WaveBody_0000201640.txt")

    # num_x = 1000
    # num_t = 1000
    # num_i_train = 1000
    # num_f_train = 11000
    # x = np.linspace(0, 1, num_x)
    # t = np.linspace(0, 0.2, num_t)
    # t_grid, x_grid = np.meshgrid(t, x)
    # T = t_grid.flatten()[:, None]
    # X = x_grid.flatten()[:, None]
    #
    # x_test = np.hstack((T, X))
    # print(x_test.shape, data['u'].shape)
    # print(x_test[:, 1][::1000])
    # print(data['u'])

    # Make GIF file.
    # num = 50
    # for i in range(num):
    #     index = int(float(9999 / num) * (i+1))
    #     # plt.plot(x_test[:, 1][index::2000], data['p'][0][index::2000])
    #     if data['x'][:, 9:][0].shape != data['p'][0][index::1000].shape:
    #         print(index)
    #         continue
    #     plt.plot(data['x'][:, 9:][0], data['p'][0][index::1000], 'r', label='W-PINN-DE')
    #     plt.plot(solution[:, 0], solution[:, -2], 'black', label='Exact')
    #     plt.legend(loc='best')
    #     plt.xlim(-0.4, 1.2)
    #     plt.ylim(-0.4, 1.2)
    #     plt.savefig(f'./work/shock_tube/gif/Sod_Shock_Tube_Predict_pinn_p-{i+1}.png')
    #     plt.clf()

    # plt.plot(data['x'][:, 9:][0], data['p'][0][9999::1000])
    # plt.xlim(-0.4, 1.2)
    # plt.show()

    ############################################################
    # print(x_sph)
    # num_x_sph = 1000
    # num_t_sph = 1000
    # num_i_train_sph = 1000
    # num_f_train_sph = 11000
    # x_sph = np.linspace(0, 1, num_x_sph)
    # t_sph = np.linspace(0, 0.2, num_t_sph)
    # t_grid_sph, x_grid_sph = np.meshgrid(t_sph, x_sph)
    # T_sph = t_grid.flatten()[:, None]
    # X_sph = x_grid.flatten()[:, None]
    #
    # x_test_sph = np.hstack((T_sph, X_sph))
    ############################################################
