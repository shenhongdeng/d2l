import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import torchvision
import numpy as np
from torch import optim
import matplotlib.pyplot as plt

dtype = torch.FloatTensor
writer = SummaryWriter(log_dir="../data/log", comment="Loss")
np.random.seed(100)
x_train = np.linspace(-1, 1, 100).reshape(100, 1)
y_train = 3 * np.power(x_train, 2) + 2 + 0.2 * np.random.rand(x_train.size).reshape(100, 1)

# model = nn.Linear(1, 1)

'''
model = torch.nn.Sequential(
    torch.nn.Linear(in_features=1, out_features=1),
    torch.nn.ReLU(),
    # torch.nn.Linear(1, 1),
)
'''


class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 Module 的 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature,
                                      n_hidden)  # 隐藏层线性输出, type(hidden) = torch.nn.modules.linear.Linear(一个类)
        self.predict = torch.nn.Linear(n_hidden,
                                       n_output)  # 输出层线性输出, type(predict) = torch.nn.modules.linear.Linear(一个类)

    def forward(self, x):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值) self.hidden.forward(x)
        x = self.predict(x)  # 输出值 self.predict.forward(x)
        return x

model = Net(n_feature=1, n_hidden=10, n_output=1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

input = torch.from_numpy(x_train).type(dtype)
targets = torch.from_numpy(y_train).type(dtype)

for epoch in range(100):
    output = model(input)
    loss = criterion(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar("训练损失值", loss.item(), epoch)
    print(loss.item())



pred = model(input)
plt.plot(input.view(1, -1)[0].detach().numpy(), targets.view(1, -1)[0].detach().numpy(), label="True")
plt.plot(input.view(1, -1)[0].detach().numpy(), pred.view(1, -1)[0].detach().numpy(), label="pred")
plt.legend()
plt.show()






