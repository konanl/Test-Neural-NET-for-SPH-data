import torch
import h5py
import numpy as np
import os
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from visdom import Visdom
import time

# Data OPerate
# 获取所有的文件
files = os.listdir('../Datasets_yyh0083_0du_9p8')
files.sort()

MAX = 150           # 划分训练集和测试集的指标, 即:训练集150,测试集44(一共194个数据集)
i = 0               # 划分训练集和测试集的初始化i

DATA_train = []
DATA_test  = []

for file in files:

    filename = os.path.splitext(file)
    if filename[1] == '.hdf5':
        # print(file)

        # 获取文件路径
        path = os.path.join('../Datasets_yyh0083_0du_9p8/' + file)
        # print(path)

        f = h5py.File(path)

        data = f['dataset']
        
#         DATA.append(data)

        if i < MAX:
            DATA_train.append(data) # 将速度作为神经网络的输出
        else:
            DATA_test.append(data)
        i += 1  # 累加i

x_train_data_PRE = np.array(DATA_train[:-1], dtype=np.float32)
y_train_data_PRE = np.array(DATA_train[1:], dtype=np.float32)

x_test_data_PRE = np.array(DATA_test[:-1], dtype=np.float32)
y_test_data_PRE = np.array(DATA_test[1:], dtype=np.float32)

# 将速度作为输出
x_train_data_v = x_train_data_PRE[:,:,-3:]
y_train_data_v = y_train_data_PRE[:,:,-3:]

x_test_data_v = x_test_data_PRE[:,:,-3:]
y_test_data_v = y_test_data_PRE[:,:,-3:]

x_train_data_v = torch.from_numpy(x_train_data_v)
y_train_data_v = torch.from_numpy(y_train_data_v)
x_test_data_v = torch.from_numpy(x_test_data_v)
y_test_data_v = torch.from_numpy(y_test_data_v)

# norm function
def data_norm(data):
    mean = data.mean(dim=2, keepdim=True)
    std = data.std(dim=2, keepdim=True)
    return (data-mean)/std

x_train_data_v_norm = data_norm(x_train_data_v)
y_train_data_v_norm = data_norm(y_train_data_v)

x_test_data_v_norm = data_norm(x_test_data_v)
y_test_data_v_norm = data_norm(y_test_data_v)

# 数据处理--Dataloader
class MyDataset(Dataset):
    def __init__(self, x, label):
        self.x = x
        self.label = label
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, item):
        return self.x[item], self.label[item]

train_data = MyDataset(x_train_data_v, y_train_data_v)

test_data = MyDataset(x_test_data_v, y_test_data_v)

# Dataloader
dataloader_train = DataLoader(train_data, batch_size=10, shuffle=True)

dataloader_test = DataLoader(test_data, batch_size=10, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# network model
class MyNetwork(nn.Module):
    def __init__(self, p_num):
        super(MyNetwork,self).__init__()  # p_num 为系统的粒子个数
        self.p_num = p_num
        self.flatten = nn.Flatten()  # 默认为（start_dim = 1, end_dim=-1）
        # flatten之后tensor的shape为(batch_size, p_num * 3)
        self.subnet_unit = nn.Sequential(nn.Linear(self.p_num*3, 120),
                                         nn.ReLU(), 
                                         nn.Linear(120, 120),
                                         nn.ReLU(), 
                                         nn.Linear(120, 100), 
                                         nn.ReLU(),
                                         nn.Linear(100, self.p_num*3))
        # self.num_subnet = num_subnet
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.subnet_unit(x)
        return x

p_num = len(x[0])

model = MyNetwork(p_num).to(device)

loss_fn = nn.MSELoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([[0.0, 1.0]], [0.], win='test', opts=dict(title='test_loss&acc.', legend=['loss', 'acc.']))

global_step = 0. # 记录训练步长

train_data = dataloader_train
test_data = dataloader_test

epochs = 100

# train
for epoch in range(epochs):
    model.train()  # train
    for batch_idx, (x, label) in enumerate(dataloader_train):
        x, label = x.to(device), label.to(device)
        y_hat = model(x)
        label = torch.flatten(label, 1, -1)
        train_loss = loss_fn(y_hat, label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        global_step += 1
        viz.line([train_loss.item()], [global_step], win='train_loss', update='append')
    if batch_idx % 100 == 0:    
        print(epoch, 'loss: {:.2f}'.format(loss.item()))
    model.eval()   # test
    with torch.no_grad():
        correct = 0
        test_loss = 0
        total_correct = 0
        total_num = 0
        for x, label in dataloader_test:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            label = torch.flatten(label, 1, -1)
            test_loss += loss_fn(logits, label).item()
            if loss < 0.5:
                correct += 1
            # correct = torch.eq(pred,label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)   
        test_loss /= total_num
        acc = total_correct / total_num
        # visdom 可视化
        viz.line([[test_loss, acc]], [global_step], win='test', update='append')
        if batch_idx % 100 == 0:
            print(epoch, 'test acc: {:.2f}%'.format(acc*100))
            print('test loss: {:.2f}'.format(loss))