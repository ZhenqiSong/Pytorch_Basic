import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim import optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 定义一个卷积层结构
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # 全连接
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积后接激活接最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # 将2d拉伸为1d
        x = x.view(-1, self.num_flat_features(x))

        # 全连接接激活
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = Net()
    print(net)

    # 获取所有参数
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    # 生成随机输入，进行前馈
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)

    # 清空梯度，使用随机梯度反向传播
    # net.zero_grad()
    # out.backward(torch.randn(1, 10))

    # # 反向传播前后对比
    # print(params[0])
    # params2 = list(net.parameters())
    # print(params[0])

    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()

    loss = criterion(out, target)
    print(loss)

    print(loss.grad_fn)
    print(loss.grad_fn.next_functions[0][0])
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

    net.zero_grad()
    print(net.conv1.bias.grad)
    loss.backward()
    print(net.conv1.bias.grad)

    learning_rate = 0.01
    if f in net.parameters():
        f.data.sub_(f.grad * learning_rate)

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    optim.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
