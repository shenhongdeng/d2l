import torch
import numpy as np
print("Support cuda ?: ", torch.cuda.is_available())

"""
广播机制
每一个维度上都会增加
"""
A = np.arange(0, 40, 10).reshape(4, 1)
B = np.arange(0, 3)
print(A)
print(B)
A = torch.from_numpy(A)
B = torch.from_numpy(B)
print(A + B)

# 二维矩阵乘法
x = torch.randint(10, (2, 3))
y = torch.randint(6, (3, 4))
print(x, y)
print(torch.mm(x, y))

# 三维矩阵的乘法
x = torch.randint(10, (2, 2, 3))
y = torch.randint(6, (2, 3, 4))
z = torch.bmm(x, y)
print(x.shape, y.shape, z.shape)