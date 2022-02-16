import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


train_batchsize = 4
test_batchsize = 4
num_works = 2
lr = 0.01
num_epoch = 5
writer = SummaryWriter('../data/log', comment="show CRF10")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = CIFAR10("../data", train=True, transform=transform, download=False)
test_dataset = CIFAR10("../data", train=False, transform=transform, download=False)

train_loader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=num_works)
test_loader = DataLoader(test_dataset, batch_size=test_batchsize, shuffle=False, num_workers=num_works)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(in_features=1296, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 36 * 6 * 6)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


# train_iter = iter(train_loader)
# images, labels = next(train_iter)
# print(len(images))

model = Net()
model = model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
train_losses = []
train_acces = []
test_acces = []
test_losser = []
model.to(device)
# 训练模型

for epoch in range(num_epoch):
    model.train()
    train_loss = 0
    train_acc = 0
    for img, label in train_loader:
        # print("A", label.shape)
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        # print("out", out.shape)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        correct = (pred == label).sum().item()
        acc = correct / img.shape[0]
        train_acc += acc
    train_acces.append(train_acc / len(train_loader))
    train_losses.append(train_loss /len(train_loader))
    print("train loss: {:.4f}, train acc: {:.4f}".format(train_loss / len(train_loader), train_acc / len(train_loader)))
    writer.add_scalar("CRF10 loss", train_loss, epoch)
    writer.add_scalar("CRF10 acc", train_acc, epoch)

    test_acc = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = criterion(out, label)
            test_loss += loss
            _, pred = out.max(1)
            correct = (pred == label).sum().item()
            acc = correct / img.size(0)
            test_acc += acc
        test_acc = test_acc / len(test_loader)
        test_loss = test_loss / len(test_loader)
        print("test loss: {:.4f}, test acc: {:.4f}".format(test_loss, test_acc))
        test_acces.append(test_acc)
        test_losser.append(test_loss)
        writer.add_scalar("CRF10 test loss", test_loss, epoch)
        writer.add_scalar("CRF10 test acc", test_acc, epoch)
writer.close()








