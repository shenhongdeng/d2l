import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch import optim

lr = 0.01
batch_size = 32
num_epoches = 12
torch.manual_seed(10)

x = torch.unsqueeze(torch.linspace(1, -1, 1000), dim=1)
y = torch.pow(x, 2) + 0.01 * torch.randn(x.size())
# 用生成的数据产生dataset对象
dataset = Data.TensorDataset(x, y)
trian_loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Net(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super(Net, self).__init__()
        self.hidden = nn.Linear(in_dim, hidden)
        self.predict = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out = F.relu(self.hidden(x))
        out = self.predict(out)
        return out

net_SGD      = Net(1, 20, 1)
net_Momentum = Net(1, 20, 1)
net_RMSProb  = Net(1, 20, 1)
net_Adam     = Net(1, 20, 1)

nets = [net_SGD, net_Momentum, net_RMSProb, net_Adam]
SGD = optim.SGD(net_SGD.parameters(), lr=lr)
Momentum = optim.SGD(net_Momentum.parameters(), lr=lr, momentum=0.9)
RMSProb = optim.RMSprop(net_RMSProb.parameters(), lr=lr, alpha=0.9)
Adam = optim.Adam(net_Adam.parameters(), lr=lr, betas=(0.9, 0.99))
optimizers = [SGD, Momentum, RMSProb, Adam]

# train model
loss_func = nn.MSELoss()
loss_his = [[], [], [], []]
for epoch in range(num_epoches):
    for step, (batch_x, batch_y) in enumerate(trian_loader):
        # 一一对应然后打包
        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(batch_x)
            loss = loss_func(output, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())
            # print(l_his)
labels = ["SGD", "Momentum", "RMSProb", "Adam"]
for i, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labels[i])
plt.legend(loc="best")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.ylim((0, 0.2))
plt.show()














