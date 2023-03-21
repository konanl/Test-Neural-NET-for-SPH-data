#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PINN_sph 
@File    ：test_shock_tube.py
@Author  ：LiangL. Yan
@Date    ：2023/2/21 21:37
"""
import datetime

import numpy as np
import torch

from model import *
from utils import *
import random
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# Seeds
torch.manual_seed(142589)
np.random.seed(142589)

# Create directories if not exist.
log_dir = './work/loss/shock_tube'
model_save_dir = "./model/shock_tube"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 一维激波管 #
# du / dt + df / dx = 0
# u = (rho, rho * u, E)
# f = (rho * u, rho * u**2 + p, (E + p) * u)

## 化简 -->
"""
    dU / dt + A * dU / dx = 0   (1)         (x, t) in (0, 1)x(0, 0.2+)
    ==>
    U = [ rho ]         A = [ u, rho, 0 ]
        [ u ]               [ 0, u, 1/rho ]
        [ p ]               [0, gamma*p, u]
"""

# params
gamma = 1.4


class ShockTubeSodPINN(PINNs):
    """The PINN model to solve the shock tube(Sod problem) which is a basic example in SPH."""
    def __init__(self, basic_net, optim_=optim.Adam):
        super(ShockTubeSodPINN, self).__init__(basic_net, optim_)

    def equation(self, X):
        """PDE."""
        x = X[:, 0:1]
        t = X[:, 1:2]

        # 网络输入为 x, t;输出为 rho, u, p
        y = self.net(X)
        rho = y[:, 0:1]
        u = y[:, 1:2]
        p = y[:, 2:3]

        # 控制方程
        # 1. du / dt + df / dx = 0
        # f1 --- drho_dt + rho * du_dx + u * drho_dx = 0
        dr = gradients(X, rho)
        du = gradients(X, u)

        drdt = dr[:, 1:2]
        drdx = dr[:, 0:1]
        dudx = du[:, 0:1]

        L_f1 = drdt + rho * dudx + u * drdx

        # f2 --- rho * (du_dt + u * du_dx) + dp_dx
        dp = gradients(X, p)

        dpdx = dp[:, 0:1]
        dudt = du[:, 1:2]

        L_f2 = rho * (dudt + u * dudx) + dpdx

        # f3 --- dp_dt + gamma * p * du_dx + u * dp_dx
        dpdt = dp[:, 1:2]

        L_f3 = dpdt + gamma * p * dudx + u * dpdx

        L_f = torch.mean(torch.square(L_f1**2)) + torch.mean(torch.square(L_f2**2)) + torch.mean(torch.square(L_f3**2))

        return L_f

    def loss_ic(self, x):
        """Loss of pde init condition."""
        x_ic = x[0]
        y_ic = self.net(x_ic)
        rho_ic, u_ic, p_ic = x[1], x[2], x[3]

        rho_ic_nn, u_ic_nn, p_ic_nn = y_ic[:, 0], y_ic[:, 1], y_ic[:, 2]

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

    for i in range(N):
        if x[i] <= 0.5:
            rho_init[i] = 1.0
            p_init[i] = 1.0
        else:
            rho_init[i] = 0.125
            p_init[i] = 0.1

    return rho_init, u_init, p_init


def gen_traindata(num):
    """Generate the train data for One-Dim Shock Tube."""
    num_x = num
    num_t = num * 2
    num_i_train = num
    num_f_train = 11000 # num * num + num // 10
    x = np.linspace(0, 1, num_x)
    t = np.linspace(0, 0.2, num_t)
    t_grid, x_grid = np.meshgrid(t, x)
    T = t_grid.flatten()[:, None]
    X = x_grid.flatten()[:, None]

    id_ic = np.random.choice(num_x, num_i_train, replace=False)
    id_f = np.random.choice(num_x * num_t, num_f_train, replace=False)

    x_ic = x_grid[id_ic, 0][:, None]
    t_ic = t_grid[id_ic, 0][:, None]
    x_ic_train = np.hstack((x_ic, t_ic))
    rho_ic_train, u_ic_train, p_ic_train = IC(x_ic)

    x_int = X[:, 0][id_f][:, None]
    t_int = T[:, 0][id_f][:, None]
    x_int_train = np.hstack((x_int, t_int))
    x_test = np.hstack((X, T))

    return x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train, x_test


def get_loader(data, mode='train', batch_size=32, num_workers=1, pack=False):
    """Build and return a data loader."""
    if pack:
        dataset = DatasetShockTube(data)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=(mode == 'train'),
                                 num_workers=num_workers)
    else:
        # [x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train]
        x_ic_train = torch.tensor(data[0], dtype=torch.float32).to(device)
        x_int_train = torch.tensor(data[1], requires_grad=True, dtype=torch.float32).to(device)
        # x_test = torch.tensor(x_test, dtype=torch.float32).to(device)

        rho_ic_train = torch.tensor(data[2], dtype=torch.float32).to(device)
        u_ic_train = torch.tensor(data[3], dtype=torch.float32).to(device)
        p_ic_train = torch.tensor(data[4], dtype=torch.float32).to(device)

        data_loader = [x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train]
    return data_loader


class DatasetShockTube(Dataset):
    """Dataset for Shock Tube."""

    def __init__(self, data):
        super(DatasetShockTube, self).__init__()

        self.data = data

        self.x_ic_train = torch.tensor(data[0], dtype=torch.float32)
        self.x_int_train = torch.tensor(data[1], dtype=torch.float32)

        self.rho_ic_train = torch.tensor(data[2], dtype=torch.float32)
        self.u_ic_train = torch.tensor(data[3], dtype=torch.float32)
        self.p_ic_train = torch.tensor(data[4], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.x_ic_train[item], self.x_int_train[item], \
               self.rho_ic_train[item], self.u_ic_train[item], self.p_ic_train


def loss(x, model):
    """Loss Function."""
    EQLoss_f, EQLoss_ic = model.equation(x[0]), model.loss_ic(x[1:])
    # w_ic, w_f = 10, 0.1
    # EQLoss = w_ic * torch.mean(torch.square(EQLoss_ic)) + w_f * torch.mean(torch.square(EQLoss_f))
    return EQLoss_f, EQLoss_ic


def train(data_loader, model, num_epochs, lr, loader_enable=False):
    """Train the model.(简化版)"""
    model.train()
    optimizer = optim.Adam(params=model.net.parameters(), lr=lr, betas=(0.9, 0.999))
    # optimizer = optim.SGD(params=model.net.parameters(), lr=lr)
    start_epoch = 0
    print("\nStart Train .....\n")
    start_time = time.time()
    with tqdm(range(start_epoch, num_epochs)) as tepochs:
        for epoch in tepochs:
            # Set the begin.
            tepochs.set_description(f"Epoch {epoch + 1}")

            # train data
            if loader_enable:
                data_iter = iter(data_loader)
                x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train = next(data_iter)
                x_int_train.requires_grad = True
            else:
                x_int_train = data_loader[1]
                x_ic_train, rho_ic_train, u_ic_train, p_ic_train = data_loader[0], data_loader[2], \
                                                                   data_loader[3], data_loader[4]

            x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train = x_ic_train.to(device), \
                                                                            x_int_train.to(device), \
                                                                            rho_ic_train.to(device), \
                                                                            u_ic_train.to(device), \
                                                                            p_ic_train.to(device)

            # Compute loss
            loss_sum = 0

            # EQLoss_f, EQLoss_ic = loss([x_int_train, x_ic_train, rho_ic_train, u_ic_train, p_ic_train], model)
            EQLoss_f, EQLoss_ic = model.equation(x_int_train), model.loss_ic([x_ic_train,
                                                                              rho_ic_train,
                                                                              u_ic_train,
                                                                              p_ic_train])

            w_ic, w_f = 10, 1
            EQLoss = w_ic * EQLoss_ic + w_f * EQLoss_f

            EQLoss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging.
            loss_ = {}
            loss_sum += EQLoss
            loss_['PINNs/loss'] = EQLoss.item()
            loss_['PINNs/loss_f'] = np.mean(np.square(EQLoss_f.item()))
            loss_['PINNs/loss_ic'] = np.mean(np.square(EQLoss_ic.item()))

            # Save loss information
            log_step = 2000
            # log_dir = './work/loss'
            if (epoch + 1) % log_step == 0:
                loss_save_path = os.path.join(
                    log_dir,
                    "{}-{}".format(
                        "NN", epoch + 1
                    )
                )
                torch.save(loss_, loss_save_path)

            # Print out training information.
            if (epoch + 1) % (log_step // 10) == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "\nElapsed [{}], Iteration [{}/{}]".format(et, epoch + 1, num_epochs)
                for tag, value in loss_.items():
                    log += ", {}: {:.2e}".format(tag, value)

                print(log)

            # Save model checkpoints.
            model_save_step = 2000
            # model_save_dir = "./model"
            if (epoch + 1) % model_save_step == 0:
                path = os.path.join(model_save_dir, '{}-pinn.ckpt'.format(epoch + 1))
                torch.save(model.net.state_dict(), path)
                print('Saved model checkpoints into {}...'.format(model_save_dir))

            # Decay learning rate...
            ##
            MLoss = torch.mean(loss_sum)
            tepochs.set_postfix(MeanLoss=MLoss.item())
            time.sleep(0.0001)

        # Save parameters information
        ##


def inference(X, model, resume_epochs, model_save_dir):
    model.eval()
    """Predicting trained model."""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        # loading model
    restore_model(model, resume_epochs, model_save_dir)
    y_pred = model(X)
    return y_pred


def main(num_epochs=6000, lr=1e-3, test=False):
    """Train main function."""

    # generate data
    x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train, x_test = gen_traindata(1000)

    # build dataloader with torch.utils.data.Dataloader
    # data_loader = get_loader([x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train], batch_size=1)

    # build dataloader with numpy and convert to Tensor
    data_loader = get_loader([x_ic_train, x_int_train, rho_ic_train, u_ic_train, p_ic_train])

    # train
    ## training params

    if not test:
        sys.stdout = Logger(
            os.path.join(
                log_dir,
                'train-{}-{}.log'.format(
                    # "NN" if config.net_type == 'pinn' else "gNN, w={}".format(config.g_weight),
                    lr,
                    num_epochs,
                )
            ),
            sys.stdout
        )

    # model
    layers = [2] + [30] * 7 + [3]
    net = FCNet(layers)
    thock_tube_model = ShockTubeSodPINN(net).to(device)
    print("The model is build like:\n")
    summary(thock_tube_model)

    if not test:
        train(data_loader, thock_tube_model, num_epochs, lr)

    if test:
        # predict
        # import matplotlib.pyplot as plt

        a = np.loadtxt("./data/shock_tube/SPHBody_WaveBody_0000201640.txt")
        x_test = a[:, 0:1]
        t_test = np.array(np.linspace(0.2, 0.2, 4000).reshape((-1, 1)), dtype=np.float32)
        input_test = np.hstack((x_test, t_test))
        input_test = torch.tensor(input_test, dtype=torch.float32).to(device)

        # x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_pred = inference(input_test, thock_tube_model, 6000, "./model/shock_tube")

        plt.title("PINN Predict")
        plt.xlabel("x")
        # plt.plot(x_test.cpu().detach().numpy()[:, 0:1], y_pred.cpu().detach().numpy()[:, 1:2], ":", label='velocity')
        plt.plot(x_test, y_pred.cpu().detach().numpy()[:, 1:2], ":", label='u')
        plt.plot(x_test, y_pred.cpu().detach().numpy()[:, 2:3], label='p')
        plt.plot(x_test, y_pred.cpu().detach().numpy()[:, 0:1], "--", label='rho')
        plt.legend(loc='best')

        # save name.
        date_time = datetime.datetime.now().strftime("%Y_%m_%d")

        plt.savefig('./work/shock_tube/predict_{}.png'.format(date_time))
        plt.show()


if __name__ == '__main__':
    # x,y,z,ID,Velocity_x,Velocity_y,Velocity_z,Pressure,TotalEnergy

    # Device
    device = get_default_device()
    print("Currently available resources: {}".format(device))

    main(test=True)
