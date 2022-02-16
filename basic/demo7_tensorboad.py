import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.liner1 = nn.Linear(320, 50)
        self.liner2 = nn.Linear(50, 10)
        self.bn = nn.BatchNorm2d(20)

    def forward(self, x):
         x = F.max_pool2d(self.conv1(x), 2)
         x = F.relu(x) + F.relu(-x)
         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
         x = self.bn(x)
         x = x.view(-1, 320)
         x = F.relu(self.liner1(x))
         x = F.dropout(x, training=self.training)
         x = self.liner2(x)
         x = F.softmax(x, dim=1)
         return x


input = torch.rand(32, 1, 28, 28)
model = Net()
with SummaryWriter(log_dir="../data/log", comment="Net") as w:
    w.add_graph(model, (input, ))



