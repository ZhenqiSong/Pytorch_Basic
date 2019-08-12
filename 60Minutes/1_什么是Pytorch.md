# 什么是Pytorch

Pytorch是一个python的科学计算包，目标人群有两类：
- 代替numpy，从而可以使用GPU的算力
- DL研究平台，提供最大的灵活性和速度

## 开始

### Tensors
**Tensors**类似numpy中的**ndarrays**,此外它支持GPU运算

```python
# 使用Tensors，需要引用 torch 包
from __future__ import print_function
import torch
```
- 创建一个随机未初始化的空tensor, 得到的值未知，不能代替随机初始化
```python
x = torch.empty(5, 3)
print(x)
```
```
out:
 tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],  
        [0.0000e+00, 0.0000e+00, 0.0000e+00],  
        [0.0000e+00, 0.0000e+00, 0.0000e+00],  
        [0.0000e+00, 1.7040e-42, 0.0000e+00],  
        [0.0000e+00, 0.0000e+00, 0.0000e+00]]) 
```
- 随机初始化的矩阵
```python
x = torch.rand(5, 3)
print(x)
```
从输出结果看，结果和未初始化的差别还是很大的
```
tensor([[0.0168, 0.4280, 0.1704],
        [0.4205, 0.0452, 0.5909],
        [0.0661, 0.6218, 0.0300],
        [0.0850, 0.6206, 0.4926],
        [0.3288, 0.7670, 0.5476]])
```
- 创建一个全0矩阵，并指定数据类型
```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```
```
out:
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
```
- 直接通过数据创建张量
```python
x = torch.tensor([5.5, 3])
print(x)
```
```
out:
tensor([5.5000, 3.0000])
```
- 同样可以通过现有张量创建，如果不指定将继承现有张量的属性
```python
# 通过现有张量创建
# 通过对象的方法new_*创建
x = x.new_ones(5, 3, dtype=torch.double)
y = x.new_zeros(5, 3)
x = torch.randn_like(x, dtype=torch.float)
```
### 操作
张量操作由许多不同的语法实现，以加法为例：
- 加法1：直接使用加号
```python
y = torch.rand(5, 3)
print(x + y)
```
- 加法2：使用方法
```python
print(torch.add(x, y))
```
- 加法3：使用参数接收结果
```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
```
- 加法4：使用Inplace，使用这种方法和直接替换掉了调用方法的变量
```python
y.add_(x)
print(y)
```
```
out:
tensor([[ 3.2936,  1.3157,  0.9915],
        [ 2.3024,  1.6349, -0.0656],
        [ 1.8533,  0.3392, -0.0429],
        [-0.6015,  0.5819,  0.1870],
        [ 0.2772, -0.4061, -0.4227]])
```
- resize: torch中的变换尺寸，使用`torch.view`
```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1 表示该尺寸由其他的尺寸计算得到
print(x.size(), y.size(), z.size())
```
```
torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```
- item: 当张量只有一个元素时，可以使用Item查看它的值,只能在张量只包含一个元素时使用
```python
x = torch.randn(1)
print(x.item())
```
```
-0.5609784126281738
```
> 更多操作[https://pytorch.org/docs/stable/torch.html]

### 与Numby
torch和numpy之间可以互相转换，前提时torch在cpu上
- 从Tensor到Numpy
```python
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)
```
```
从结果上看，其实torch和numpy的数据共享同一块内存
tensor([2., 2., 2., 2., 2.])
[2. 2. 2. 2. 2.]
```
- 从Numpy到Tensor
```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
```
## CUDA 张量
使用`.to`方法可以把张量移到任何设备
```python
# 首先判断是否有cuda设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double)) # 使用to方法时，同样可以指定数据类型
```