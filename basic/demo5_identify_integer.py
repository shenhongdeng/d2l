import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 设置学习的基本参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 5
lr = 0.01
momentum = 0.5 # SGD中的动量大小

# 定义数据转换，Compose是将两个结合起来，顺序执行，Normalize是对张量进行归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = MNIST("../data/integers", train=True, transform=transform, download=False)
test_dataset  = MNIST("../data/integers", train=False, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)


# 显示部分的模型
examples = enumerate(test_loader)
# batch_idx为是哪一个batch的索引值，后边的a_batch指的是具体的值一个batch包含的内容，由两个元素，第一个存放的是128张图片，第二个存放的128个标签
batch_idx, a_batch = next(examples)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    # 因为每一张图片只有1个通道，所以我们用a_batch[0][i][0]就可以得到这个通道的输出
    plt.imshow(a_batch[0][i][0], cmap="gray", interpolation="none")
    plt.title("Ground Truth: {}".format(a_batch[1][i]))
    plt.xticks()
    plt.yticks()
plt.show()

# 定义模型类
class Net(nn.Module):
    def __init__(self, in_dim, hidden_1, hidden_2, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_1), nn.BatchNorm1d(hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(hidden_1, hidden_2), nn.BatchNorm1d(hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(hidden_2, out_dim))

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net(28 * 28, 300, 100, 10)
model.to(device)
# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
for epoch in range(num_epoches):
    train_loss = 0
    train_acces = 0
    model.train()
    # 动态更新参数，因为优化器中有动量，所以我们如果直接重新整一个全新的优化器的话，动量会损失
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1
    # img中是64张图片
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        # print(img.shape)
        # 这里直接把图片展开
        img = img.view(img.size(0), -1)
        # out分到每一个数字概率，是一个64*10的二维数组
        out = model(img)
        # print(out.size())
        # 这个loss的数据类型tensor(2.3588, grad_fn=<NllLossBackward>)，需要我们用item来获取
        loss = criterion(out, label)
        # print(loss)
        # 反向传播之前需要先把梯度清零
        optimizer.zero_grad()
        # 开始更新数据
        loss.backward()
        optimizer.step()
        # .item不只是把tensor的数拿出来，发现只是拿出维度为（1,1）的数字，转换成标量 常用于计算损失的时候
        train_loss += loss.item()
        # print(train_loss)
        # print(out.max(0))
        # out.max(1)是返回每行中最大的值，0返回的是每一列中最大的值
        # out.max()返回两个值，第一个是最大的值，第二个是返回的最大的值的index
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        # print(num_correct)
        acc = num_correct / label.shape[0]
        train_acces += acc
    losses.append(train_loss / len(train_loader))
    acces.append(train_acces / len(train_loader))
    eval_loss = 0
    eval_acc  = 0
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.shape[0], -1)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        acc = (pred == label).sum().item() / img.shape[0]
        eval_acc += acc
    eval_acces.append(eval_acc / len(test_loader))
    eval_losses.append(eval_loss / len(test_loader))
    print("epoch: {:.4f}, Train losses: {:.4f}, Train acc: {:.4f}, Test losses: {:.4f}, Test acc: {:.4f}"
          .format(epoch, train_loss / len(train_loader), train_acces / len(train_loader), eval_loss / len(test_loader), eval_acc / len(test_loader)))

torch.save(model, "../data/model/identify_integer.pth")
torch.save(model.state_dict(), "../data/model/identify_integer_param.pth")




plt.title("trianloss")
plt.plot(np.arange(num_epoches), losses, label="train_loss")
plt.plot(np.arange(num_epoches), acces, label="train_acc")
plt.legend()
plt.show()
















