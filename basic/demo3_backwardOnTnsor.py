import torch

x = torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)
J = torch.zeros(2, 2)
print(J)
y = torch.zeros(1, 2)
print(y)
y[0, 0] = x[0, 0] ** 2 + x[0, 1] * 3
y[0, 1] = x[0, 1] ** 2 + x[0, 0] * 2
print(y)

y.backward(torch.tensor([[1, 1]]))
print("y.grad:", y.grad)
print("x.grad:", x.grad)
print(torch.mul(x.grad, torch.t(torch.tensor([[1, 1]]))))
print(torch.t(torch.tensor([[1, 1]])).shape)
# 参考教材41页


