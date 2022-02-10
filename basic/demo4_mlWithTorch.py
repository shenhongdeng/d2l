import torch
import matplotlib.pyplot as plt

# 设计随机数种子
torch.manual_seed(100)
dtype = torch.float
x = torch.linspace(-1, 1, 100)
# 这里dim=1是shape里边的第二个维度
x = torch.unsqueeze(x, dim=1)
# 在torch中".shape"和".size()"的效果是一样的
y = 3 * torch.pow(x, 2) + 2  + 0.2* torch.rand(x.shape)
# 这里要转化成为numpy才能输出
plt.scatter(x.numpy(), y.numpy())
plt.show()

lr = 0.001
w = torch.randn(1, 1, dtype=dtype, requires_grad=True)
b = torch.randn(1, 1, dtype=dtype, requires_grad=True)
for i in range(8000):
    predict = w * torch.pow(x, 2) + b
    loss = 0.5 * torch.pow(predict - y, 2)
    loss = loss.sum()
    loss.backward()
    # torch.no_grad()，使上下文中切断自动求导的计算
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()
plt.plot(x.numpy(), y.numpy(), color="g", label="true")
# detach传播到这里就会停止传播，这里相当与把tensor的require_grad设置为false。
plt.scatter(x.numpy(), (w * torch.pow(x, 2) + b).detach().numpy(), color="r", label="predict")
plt.legend()
plt.show()
print((w * torch.pow(x, 2) + b).detach().requires_grad)
print(w, b)
