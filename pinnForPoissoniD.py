#!/usr/bin/env python
# encoding: utf-8
"""
    @author: LiangL. Yan
    @license: (C) Copyright forever
    @contact: yokingyan@gmail.com
    @ide: pycharm
    @file: poisson 1d.py
    @time: 2022/10/3 20:43
    @desc:
    
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import visual_data
from visualization import *
from pde import *
from solver import *
from model import *
from process_data import *

pi = np.pi


def get_config():
    parser = argparse.ArgumentParser("PINNs for 1D Poisson model", add_help=False)

    # Training configurations.
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net_type', default='gpinn', type=str)
    parser.add_argument('--epochs_adam', default=20000, type=int)
    parser.add_argument('--save_freq', default=5000, type=int, help="frequency to save model and image")
    parser.add_argument('--print_freq', default=1000, type=int, help="frequency to print loss")
    parser.add_argument('--device', default=False, type=bool, help="use gpu")

    # parser.add_argument('--num_epochs', type=int, default=20000, help='number of total iterations for training')
    # parser.add_argument('--resume_epoch', type=int, default=0, help='resume training from this step')
    # parser.add_argument('--num_epochs_decay', type=int, default=10000, help='number of iterations for decaying lr')

    # Test configuration.
    # parser.add_argument('--test_epochs', type=int, default=2000, help='how long we should test model')

    # Directories.
    parser.add_argument('--model_save_dir', type=str, default='./models/poisson-1D/model')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--log_dir', type=str, default='models/poisson-1D/logs')
    parser.add_argument('--work_name', default='Poisson-1D', type=str, help="save_path")

    # Step size.
    parser.add_argument('--log_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    # others
    parser.add_argument('--Nx_EQs', default=15, type=int)
    parser.add_argument('--Nx_Val', default=200, type=int)
    parser.add_argument('--g_weight', default=0.01, type=float)

    return parser.parse_args()


class PoissonPINNs(PINNs):
    """PINNs model of 1d poisson."""
    def __init__(self, net_):
        super(PoissonPINNs, self).__init__(net_, data=[0, 0])

    def output_transform(self, x, y):
        """Output transform."""
        return torch.tanh(x) * torch.tanh(pi - x) * y + x

    def pde(self, x):
        y = self.net(x)
        y = self.output_transform(x, y)
        dudx = gradients(x, y)
        d2udx2 = gradients(x, dudx)

        f = 8 * torch.sin(8 * x)
        for i in range(1, 5):
            f += i * torch.sin(i * x)
        eqs = f + d2udx2
        if 'gpinn' in config.net_type:
            g_eqs = gradients(x, eqs)
        else:
            g_eqs = torch.zeros((1,), dtype=torch.float32)
        return eqs, g_eqs


def train(x, BCs, y_true, model, Loss, optimizer, schedular, log_loss, config):

    inn_EQs = torch.linspace(BCs[0], BCs[1], config.Nx_EQs+2, dtype=torch.float32)[1:-1, None]
    inn_EQs = torch.tensor(inn_EQs, requires_grad=True)
    inn_DTs = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    out_DTs = torch.tensor(y_true, requires_grad=False)
    inn_BCs = torch.tensor(BCs, dtype=torch.float32)[:, None]

    def closure():

        optimizer.zero_grad()

        # inn_EQs.require_grad_(True)
        out_EQs__ = model(inn_EQs)
        out_EQs_ = model.output_transform(inn_EQs, out_EQs__)
        res_EQs, res_GEQs = model.pde(inn_EQs)

        # inn_DTs.require_grad(False)
        out_DTs__ = model(inn_DTs)
        out_DTs_ = model.output_transform(inn_DTs, out_DTs__)

        inn_BCs.requires_grad_(False)
        out_BCs__ = model(inn_BCs)
        out_BCs_ = model.output_transform(inn_BCs, out_BCs__)

        data_loss = Loss(out_DTs_, out_DTs)
        bcs_loss_1 = (out_BCs_**2).mean()
        eqs_loss = (res_EQs**2).mean()
        geqs_loss = (res_GEQs ** 2).mean()

        if config.net_type == 'pinn':
            loss_batch = eqs_loss
        else:
            loss_batch = eqs_loss + geqs_loss * config.g_weight

        loss_batch.backward()
        log_loss.append([eqs_loss.item(), geqs_loss.item(), bcs_loss_1.item(), data_loss.item()])
        return loss_batch

    optimizer.step(closure)
    schedular.step()


def inference(x, model):
    x.requires_grad = True
    y_pred = model(x)
    y_pred = model.output_transform(x, y_pred)

    y_g_pred = gradients(x, y_pred)

    return [y_pred, y_g_pred]


def gen_all(num):
    xvals = np.linspace(0, pi, num, dtype=np.float32)
    yvals = poisson_sol(xvals)
    ygrad = poisson_sol_grad(xvals)
    return np.reshape(xvals, (-1, 1)), np.reshape(yvals, (-1, 1)), np.reshape(ygrad, (-1, 1))


if __name__ == "__main__":

    config = get_config()

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    save_path = config.net_type + '-Nx_EQs_' + str(config.Nx_EQs)
    work_path = os.path.join('work', config.work_name, save_path)
    train_path = os.path.join('work', config.work_name, save_path, 'train')
    isCreated = os.path.exists(train_path)
    if not isCreated:
        os.makedirs(train_path)

    sys.stdout = visual_data.Logger(os.path.join(work_path, 'train.log'), sys.stdout)
    print(config)

    train_x, train_u, _ = gen_all(config.Nx_EQs)
    valid_x, valid_u, valid_g = gen_all(config.Nx_Val)
    BCs = [0, pi]

    # Loss
    L2Loss = nn.MSELoss()

    net = FCNet([1, ] + [20, ] * 3 + [1, ])
    Net_model = PoissonPINNs(net)

    optimizer = optim.Adam(Net_model.parameters(), lr=0.001, betas=(0.9, 0.999))
    boundary_epoch = [config.epochs_adam*6/10, config.epochs_adam*9/10]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=boundary_epoch, gamma=0.1)

    start_time = time.time()
    start_epoch = 0
    log_loss = []
    log_l2 = []

    # start_epoch, log_loss = Net_model.load_model(os.path.join(work_path, 'latest_model.pth'))

    for i in range(start_epoch):
        scheduler.step()

    # train
    for epoch in range(start_epoch, config.epochs_adam):
        train(train_x, BCs, train_u, Net_model, L2Loss, optimizer, scheduler, log_loss, config)
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        # predict
        input_test = torch.tensor(np.linspace(BCs[0], BCs[1], 1002)[:, None], dtype=torch.float32)
        output_test = poisson_sol(input_test).numpy()
        output_pred_ = inference(torch.tensor(input_test, dtype=torch.float32), Net_model)[0]
        output_pred = output_pred_.detach().numpy()

        L2_u = np.mean(np.linalg.norm(output_pred - output_test) / np.linalg.norm(output_test))

        log_l2.append([L2_u])

        if epoch > 0 and epoch % config.print_freq == 0:

            log = "Iteration [{}/{}]".format(epoch + 1, config.epochs_adam)
            log += "EQs_loss: {:.2e}".format(log_loss[-1][0])
            log += ", GEQs_loss: {:.2e}".format(log_loss[-1][1])
            log += ", BCs_loss: {:.2e}".format(log_loss[-1][2])
            log += ", DTs_loss: {:.2e}".format(log_loss[-1][3])
            print(log)

        if epoch > 0 and epoch % config.save_freq == 0:
            torch.save({'epoch': iter, 'model': Net_model.state_dict(), 'log_loss': log_loss},
                       os.path.join(work_path, 'latest_model.pth'), )

    # Plot
    plt.rcParams.update({"font.size": 16})

    # Exact u
    plt.figure(4, figsize=(20, 15))
    plt.plot(valid_x, valid_u, label="Exact", color="black")

    valid_x_ = torch.tensor(valid_x, requires_grad=True)

    u_pred = inference(valid_x_, Net_model)[0]
    plt.plot(valid_x, u_pred.detach().numpy(), label="{}, w = 0.01".format("NN" if config.net_type == "pinn" else "gNN")
             , color="red", linestyle="dashed")

    plt.xlabel("x")
    plt.ylabel("u")

    plt.legend(frameon=False)
    plt.savefig('./result/figure/poisson-1D/u.png', dpi=120)

    # Exact u`
    plt.figure(4, figsize=(20, 15))
    plt.clf()
    plt.plot(valid_x_.detach().numpy(), valid_g, label="Exact", color="black")
    u_g_pred = inference(valid_x_, Net_model)[1]
    plt.plot(valid_x_.detach().numpy(), u_g_pred.detach().numpy(), label="gNN, w = 0.01",
             color="red", linestyle="dashed")

    plt.xlabel("x")
    plt.ylabel("u`")
    plt.legend(frameon=False)
    plt.savefig('./result/figure/poisson-1D/u_g.png', dpi=120)
    plt.show()



