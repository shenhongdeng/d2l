import torch
from torch import nn

# input_size——输入的单个向量的特征数量
# hidden_size——中间隐藏状态的长度，就是直接要被下一级的使用的那个隐藏状态
# num_layers是RNN的层数，是在垂直方向上堆叠的
# batch_first，保持batch在第一位
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2)
# 这里显示用的是l不是1
print("wih(输入到隐藏层)的形状为{}, whh(影藏状态到隐藏)的形状为{}, bih(输入到影藏状态)的偏执为{}".format(
    rnn.weight_ih_l0.shape, rnn.weight_hh_l0.shape, rnn.bias_hh_l0.shape))
print("wih(输入到隐藏层)的形状为{}, whh(影藏状态到隐藏)的形状为{}, bih(输入到影藏状态)的偏执为{}".format(
    rnn.weight_ih_l1.shape, rnn.weight_hh_l1.shape, rnn.bias_hh_l1.shape))

# seq_len, batch, feature
input = torch.randn(100, 32, 10)
# layer, batch, feature_num
h_0 = torch.zeros(2, 32, 20)
output, h_1 = rnn(input, h_0)
print("output shape: {}, h_1 shape :{}".format(output.shape, h_1.shape))


# RNNCell中输入只能是一个单个的时刻，不是一个时间的序列，所以也只有一层，同时也不再有seq_len，只有batch_size和feature_num

