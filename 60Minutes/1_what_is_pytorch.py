from __future__ import print_function
import torch

# 创建一个随机未初始化的空tensor
x = torch.empty(5, 3)
print(x)

# 随机初始化的矩阵
x = torch.rand(5, 3)
print(x)