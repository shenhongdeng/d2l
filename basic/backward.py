import torch
x = torch.Tensor([2])
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
y = torch.mul(x, w)
z = torch.add(y, b)
# 查看是否具有需要计算梯度的属性
print("x.require_grad:{}, y.require_grad:{}, z.require_grad:{}, w.require_grad:{}, b.require_grad:{}".format(x.requires_grad, y.requires_grad, z.requires_grad, w.requires_grad, b.requires_grad))
# 查看是否是叶子节点
print("x.is_leaf:{}, y.is_leaf:{}, z.is_leaf:{}, w.is_leaf:{}, b.is_leaf:{}".format(x.is_leaf, y.is_leaf, z.is_leaf, w.is_leaf, b.is_leaf))
# 查看是否具有grad_fn，也就是是不是中间节点，中间节点才会具有这个属性。
print("x.grad_fn:{}, y.grad_fn:{}, z.grad_fn:{}, w.grad_fn:{}, b.grad_fn:{}".format(x.grad_fn, y.grad_fn, z.grad_fn, w.grad_fn, b.grad_fn))
print("before backward--> x.grad:{}, y.grad:{}, z.grad:{}, w.grad:{}, b.grad:{}".format(x.grad, y.grad, z.grad, w.grad, b.grad))
z.backward(retain_graph=True)
print("after backward-->x.grad:{}, y.grad:{}, z.grad:{}, w.grad:{}, b.grad:{}".format(x.grad, y.grad, z.grad, w.grad, b.grad))
# 需要多次的计算梯度的时候，我们需要把retrain_graph = True，如果我们不设置的话，在调用backward会报错
# z.backward()
z.backward()
# 这个时候梯度进行了累加
print("second time backward-->x.grad:{}, y.grad:{}, z.grad:{}, w.grad:{}, b.grad:{}".format(x.grad, y.grad, z.grad, w.grad, b.grad))
