#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PINN_sph
@File    ：test_2023_1_9.py
@Author  ：LiangL. Yan
@Date    ：2023/1/09 20:05
"""
import datetime

from model import *
from utils import *
import random
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# Device
device = get_default_device()
print("Currently available resources: {}".format(device))

# Create directories if not exist.
log_dir = './work/loss'
model_save_dir = "./model"

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


class PINNsph(PINNs):

    def __init__(self, basic_net, optim_=optim.Adam, time_depend=False):
        super(PINNsph, self).__init__(basic_net, optim_)

        # gravity
        self.v = nn.Parameter(torch.tensor([1.0, ]), requires_grad=True)
        self.k = nn.Parameter(torch.tensor([1.0, ]), requires_grad=True)

        # time dependence
        self.time_depend = time_depend

    ### 方案一：尝试用 F = GMm / R^2 计算
    #### 但是发现还是缺少光滑半径内其他粒子的插值，
    #### 还是不能很好拟合进loss function
    # def gravity(self, x, m):
    #     """
    #     F = G*Mm / R^2
    #     :return:f
    #     """
    #     f = G * self.v * m**2 / (self.k * x**2)
    #     return f

    def equation(self, X):
        # x = X[:, 0:1]
        x = X

        if self.time_depend:
            t = X[:, 1:2]
            # m = X[:, 2:3]
        else:
            t = 0
            # m = X[:, 1:2]

        r = self.net(x)[:, 0:1]
        p = self.net(x)[:, 1:2]
        v = self.net(x)[:, 2:3]

        dr_dx = gradients(x, r)[:, 0:1]
        dp_dx = gradients(x, p)[:, 0:1]
        dv_dx = gradients(x, v)[:, 0:1]

        if self.time_depend:
            dr_dt = gradients(t, r)[:, 1:2]
            dv_dt = gradients(t, v)[:, 1:2]
        else:
            dr_dt = dv_dt = 0

        # 连续性方程
        equ_c = dr_dt + dr_dx * v + dv_dx * r

        # 动量方程
        # 原因是gravity和NN输入在计算图里没有梯度关系，这里先尝试当成常数优化
        f = gravity(x)
        equ_m = r * dv_dt + r * v * dv_dx + dp_dx - gradients(x, f)[:, 0:1]
        # equ_m = r * dv_dt + r * v * dv_dx + dp_dx
        # equ = equ_c + equ_m

        return equ_c, equ_m


### 方案二：用单独的神经网络来拟合重力项再一起放进loss function里，
#### 由于，没有“带标签“的数据，所以只能做无监督学习
def gravity(X):
    input_dim = X.shape[0]
    mass = torch.tensor(np.linspace(3e-5, 5e-5, 1), dtype=torch.float32).to(device)

    # net 的输出是 F
    g_net = GravityNet(input_dim, mass).to(device)

    return g_net(X)


class DatasetSPH(Dataset):
    """Dataset for train."""

    def __init__(self, data_):
        super(DatasetSPH, self).__init__()

        self.data = data_
        self.pos = data_[:, 0:1]
        self.rho = data_[:, 1:2]
        self.vel = data_[:, 2:3]
        self.press = data_[:, 3:4]

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, index):
        return self.data[index, :]

##################
# 监督学习生成数据 #
##################
# def gen_train_data():
#     """Generate train data."""
#     file = './data/b051-v30-2.25MM-1.125IMP-MANEOS.00100'
#     bin = pynbody.tipsy.TipsySnap(file)
#
#     vel = np.array(bin['vel'], dtype=np.float32)
#     pos = np.array(bin['pos'], dtype=np.float32)  # x / NN input
#     rho = np.array(bin['rho'] * 0.368477, dtype=np.float32)
#     press = np.loadtxt("./data/b051-v30-2.25MM-1.125IMP-MANEOS.00100.press")[1:]
#     press = np.array(press, dtype=np.float32)
#
#     assert len(vel) == len(pos) & len(vel) == len(rho) & len(vel) == len(press) & len(pos) == len(rho) & len(
#         pos) == len(press) & len(rho) == len(press)
#
#     print("The size of train data is {}".format(len(vel)))
#
#     return vel, pos, rho, press


#######################
##  无监督学习生成数据  ##
#######################
# 数据范围
# vel-(-40, 40), pos-(-4, 4), rho-(0, 23), p-数值范围太大无法确定
def gen_train_data(num):
    """Generate train data."""
    pos = np.linspace(-4, 4, num)
    mass = np.linspace(3e-5, 5e-5, num)
    pos = np.array(pos, dtype=np.float32)
    mass = np.array(mass, dtype=np.float32)
    return pos, mass


def get_loader(data, mode='train', batch_size=32, num_workers=1):
    """Build and return a data loader."""
    dataset = DatasetSPH(data)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)
    return data_loader


def loss(x, model):
    """The training loss."""
    EQLoss_c, EQLoss_m = model.equation(x)
    w_c, w_m = 1, 1
    EqLoss = w_c * torch.mean(torch.square(EQLoss_c)) + w_m * torch.mean(torch.square(EQLoss_m))
    return EqLoss


def train(data_loader, model, num_epochs, lr):
    """Train the model."""
    optimizer = optim.Adam(params=model.net.parameters(), lr=lr, betas=(0.9, 0.999))
    start_epoch = 0
    print("\nStart Train .....\n")
    start_time = time.time()
    params_log = []
    with tqdm(range(start_epoch, num_epochs)) as tepochs:
        for epoch in tepochs:
            # Set the begin.
            tepochs.set_description(f"Epoch {epoch + 1}")

            # train data
            data_iter = iter(data_loader)
            x = next(data_iter)
            x.requires_grad = True
            x = x.to(device)

            # Compute loss
            loss_sum = 0
            # y_pred = model(x)
            EQLoss = loss(x, model)
            Loss = EQLoss

            Loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging.
            loss_ = {}
            loss_sum += Loss
            loss_['PINNs/loss'] = Loss.item()

            # Save loss information
            log_step = 100
            # log_dir = './work/loss'
            if (epoch + 1) % log_step == 0:
                loss_save_path = os.path.join(
                    log_dir,
                    "{}-{}".format(
                        "NN", epoch + 1
                    )
                )
                torch.save(loss, loss_save_path)

            # Print out training information.
            if (epoch + 1) % log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, epoch + 1, num_epochs)
                for tag, value in loss_.items():
                    log += ", {}: {:.2e}".format(tag, value)

                print(log)

            # Save model checkpoints.
            model_save_step = 1000
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
    """Predicting trained model."""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        # loading model
    restore_model(model, resume_epochs, model_save_dir)
    y_pred = model(X)
    return y_pred


def gen_testdata():
    import pynbody
    file = './data/b051-v30-2.25MM-1.125IMP-MANEOS.00100'
    bin = pynbody.tipsy.TipsySnap(file)
    # input
    pos = np.reshape(np.array(bin['pos'][:, 0:1], dtype=np.float32), (-1, 1))
    mass = np.reshape(np.array(bin["mass"], dtype=np.float32), (-1, 1))
    # output
    vel = np.reshape(np.array(bin["vel"][:, 0:1], dtype=np.float32), (-1, 1))
    rho = np.reshape(np.array(bin["rho"] * 0.368477, dtype=np.float32), (-1, 1))
    press = np.reshape(np.loadtxt("./data/b051-v30-2.25MM-1.125IMP-MANEOS.00100.press")[1:], (-1, 1))
    in_ = torch.tensor(np.hstack((pos, mass)), dtype=torch.float32)
    out_ = torch.tensor(np.hstack((vel, rho, press)), dtype=torch.float32)
    return in_, out_


if __name__ == '__main__':

    net = FCNet([2, 20, 20, 20, 3])
    sph_pinn_model = PINNsph(net).to(device)
    # print_network(sph_pinn_model, 'sph')

    # vel, pos, rho, press = gen_train_data()
    num_train_point = 10
    pos, mass = gen_train_data(num_train_point)

    # print(np.mean(pos[:, 0:1]), np.var(pos[:, 0:1]))
    # pos_x = norm(np.reshape(pos[:], (-1, 1)))
    pos_x = np.reshape(pos[:], (-1, 1))
    # vel_x = np.reshape(vel[:, 0:1], (-1, 1))
    # rho = np.reshape(rho, (-1, 1))
    # press = np.reshape(press, (-1, 1))
    mass = np.reshape(mass[:], (-1, 1))

    # data = np.hstack((pos_x, rho, vel_x, press))
    data = np.hstack((pos_x, mass))
    # print(np.mean(train_data[:, 0:1]), np.var(train_data[:, 0:1]))

    # x = norm(pos)[:, 0:1] # network input
    # x = [i for i in x if i >= 0 and i <= 1]
    # x = torch.tensor(x, requires_grad=True, dtype=torch.float32)

    print("After some data-preprocess shape : {}".format(data.shape))

    # train / test = 4 : 1
    random.shuffle(data)
    train_data = data[0:int(len(data) * 4 / 5), :]
    test_data = data[int(len(data) * 4 / 5):, :]
    print("train_data / test_data : {} / {} ".format(len(train_data), len(test_data)))

    # Get DataLoader
    train_dataloader = get_loader(train_data)
    test_dataset = get_loader(test_data, mode='test')

    # train
    ## training params
    num_epochs = 1000
    lr = 1e-3

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

    # Test
    #     it = iter(train_dataloader)
    #     t = next(it)
    #     tt = sph_pinn_model(t[:, 0:1])
    #     print(tt)

    # train
    # train(train_dataloader, sph_pinn_model, num_epochs, lr)

    # predict
    input_data, output_data = gen_testdata()
    input_data = input_data.to(device)
    output_data = output_data.to(device)
    out_pred = inference(input_data, sph_pinn_model, 1000, "./model")
    print(out_pred)

    # Compute loss
    loss = torch.mean(torch.abs(out_pred - output_data))
    print(loss)
    print(sph_pinn_model.v, sph_pinn_model.k)
