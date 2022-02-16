import torch
from torch import nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cell_size):
        super(LSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.gate = nn.Linear(input_size + hidden_size, cell_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax()

    def forward(self, input, cell, hidden):
        combine = torch.cat((input, hidden), dim=1)
        print(combine.shape)
        f_gate = self.sigmoid(self.gate(combine))
        i_gate = self.sigmoid(self.gate(combine))
        o_gate = self.sigmoid(self.gate(combine))
        z_state = self.tanh(self.gate(combine))
        cell = torch.add(torch.mul(cell, f_gate), torch.mul(z_state, i_gate))
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        output =self.softmax(output)
        return output, hidden, cell

    def initHidden(self):
        '''这个函数用来初始化最初的隐藏状态函数，而且只有一层'''
        return torch.zeros(1, self.hidden_size)

    def initCell(self):
        '''初始化状态，这个状态是教材中最上边的哪一个C状态'''
        return torch.zeros(1, self.cell_size)

lstmCell = LSTMCell(input_size=10, hidden_size=20, cell_size=20, output_size=10)
# 这个input是batch，feature_dim
input = torch.randn(32, 10)
cell = lstmCell.initCell()
# 输入也是batch, dim
hidden = torch.randn(32, 20)

output, hidden, cell = lstmCell(input, cell, hidden)
print(output.shape, hidden.shape, cell.shape)

