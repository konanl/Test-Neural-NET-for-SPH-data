import os
import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from torchdiffeq.torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchdiffeq import odeint


batch_time = 10
batch_size = 20
data_size = 200

viz = True

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
DATA_TYPE = 'NNET_MORE'

if viz:
    makedirs(DATA_TYPE)
    import matplotlib.pyplot as plt

def visualize(true_y, pred_y, odefunc, itr, epoch, loss):
  
    if viz:
        title = "epoch:" + str(epoch) + "           loss: " + str(loss) 
        
        plt.figure(facecolor='white')
        plt.xlabel('t')
        plt.ylabel('x, y')
        plt.title(title, loc='left')
        plt.grid()
        plt.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        plt.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        plt.savefig(DATA_TYPE + '/ts' + str(itr) + '.png')
        #plt.legend()
        plt.show()
        
        plt.figure(facecolor='white')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title, loc='left')
        plt.grid()
        plt.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        plt.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        plt.savefig(DATA_TYPE + '/phase' + str(itr) + '.png')
        #plt.legend()
        plt.show()
        

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        

def plot_(x, y):
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    
# class ODEFunc(nn.Module):

#     def __init__(self):
#         super(ODEFunc, self).__init__()

#         self.net = nn.Sequential(
#             nn.Linear(2, 50),
#             nn.Tanh(),
#             nn.Linear(50, 2),
#         )

#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)

#     def forward(self, t, y):
#         return self.net(y)


# true_y0 = torch.tensor([[1., 1.]])
# t = torch.linspace(-15., 15., data_size)


# class Lambda3(nn.Module):
  
#     def __init__(self):
#         super(Lambda3, self).__init__()
#         self.fc1 = nn.Linear(2, 25, bias = False)
#         self.fc2 = nn.Linear(25, 50, bias = False)
#         self.fc3 = nn.Linear(50, 10, bias = False)
#         self.fc4 = nn.Linear(10, 2, bias = False)
#         self.relu = nn.ELU(inplace=True)
        
#     def forward(self, t, y):
#         x = self.relu(self.fc1(y * t))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         return x
     
# with torch.no_grad():
#     true_y = odeint(Lambda3(), true_y0, t, method='dopri5')


true_y0 = torch.tensor([[1., 1.]]).to(device)
t = torch.linspace(0., 5., data_size).to(device)

class Lambda_e(nn.Module):
    
    def forward(self, t, y):
        return y
    
with torch.no_grad():
    true_y = odeint(Lambda_e(), true_y0, t, method='dopri5').to(device)
    

plot_(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0])

# plt.figure()
# plt.plot(t.numpy(), true_y.numpy()[:, ｜0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
# plt.show()device

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 150),
            nn.Tanh(),
            nn.Linear(150, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


plt.figure()
plt.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
plt.show()

plt.figure()
plt.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
plt.show()


ii = 0
niters = 1000000

func = ODEFunc().to(device)
optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
end = time.time()

time_meter = RunningAverageMeter(0.97)
loss_meter = RunningAverageMeter(0.97)

for itr in range(1, niters + 1):
  optimizer.zero_grad()
  batch_y0, batch_t, batch_y = get_batch()
  pred_y = odeint(func, batch_y0, batch_t).to(device)
  loss = torch.mean(torch.abs(pred_y - batch_y))

  loss.backward()
  optimizer.step()

  time_meter.update(time.time() - end)
  loss_meter.update(loss.item())

  if itr % 5000 == 0:
      with torch.no_grad():
          pred_y = odeint(func, true_y0, t)
          loss = torch.mean(torch.abs(pred_y - true_y))
          print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
          visualize(true_y, pred_y, func, ii, itr, round(loss.item(), 4))
          ii += 1

  end = time.time()

