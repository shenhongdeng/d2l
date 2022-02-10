import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, -1)
y = 3 * np.power(x, 2) + 2 + 0.2 * np.random.rand(x.size).reshape(100, -1)
plt.scatter(x, y)
plt.show()

w = np.random.rand(1, 1)
b = np.random.rand(1, 1)
lr = 0.001
for i in range(800):
    predict = w * np.power(x, 2) + b
    loss = 0.5 * (predict - y) ** 2
    loss = loss.sum()
    grad_w = np.sum((predict - y) * np.power(x, 2))
    grad_b = np.sum(predict - y)
    w -= lr * grad_w
    b -= lr * grad_b
print(w, b)

predict = w * np.power(x, 2) + b
plt.plot(x, y, color="g", label=True)
plt.scatter(x, predict, color="r", label=True)
plt.legend()
plt.show()



