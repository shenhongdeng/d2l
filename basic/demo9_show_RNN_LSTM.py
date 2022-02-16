import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, output_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softMax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 将输入和前一层影藏的输出连接
        combine = torch.cat((input, hidden), 1)
        hidden = self.i2h(combine)
        output = self.i2o(combine)
        output = self.softMax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(input_size=10, hidden_size=128, output_size=20)
writer = SummaryWriter('../data/log', comment="RNN")
print(rnn)

hidden = rnn.initHidden()
input = torch.randn(1, 10)
with SummaryWriter(log_dir="../data/log", comment="Net") as w:
    w.add_graph(rnn, (input, hidden))






