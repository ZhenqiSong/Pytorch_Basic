from __future__ import print_function
import torch
import numpy as np

# 创建一个随机未初始化的空tensor
x = torch.empty(5, 3)
print(x)

# 随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 全0矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

# 通过现有张量创建
# 通过对象的方法new_*创建
x = x.new_ones(5, 3, dtype=torch.double)
y = x.new_zeros(5, 3)
print(y)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# 加法
y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

# resize
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())

# item
x = torch.randn(1)
print(x.item())

# numpy互换
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# cuda张量
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
