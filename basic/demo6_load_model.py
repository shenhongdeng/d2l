import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch.nn import functional as F


test_batch = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dataset = MNIST("../data/integers", train=False, transform=transform, download=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""加载模型的时候，先把模型类拷贝过来"""
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

model = torch.load("../data/model/identify_integer.pth")
criterion = nn.CrossEntropyLoss()
model.eval()
test_loss = 0
for img, label in test_loader:
    img = img.to(device)
    label = label.to(device)
    img = img.view(img.shape[0], -1)
    out = model(img)
    loss = criterion(out, label)
    test_loss += loss
    _, pred = out.max(1)
    acc = (pred == label).sum().item() / label.shape[0]
    print(acc)


