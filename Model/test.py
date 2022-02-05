import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import MyModel


num_epochs = 2
train_size = 200
test_size = 1000
lr = 0.05
momentum = 0.5#动量梯度下降法
random_seed = 1
torch.manual_seed(random_seed)

# 读取数据集
train_iter = DataLoader(
    torchvision.datasets.MNIST(
        root='../data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=train_size,
    shuffle=True,
)
test_iter = DataLoader(
    torchvision.datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=test_size,
    shuffle=True,
)
# print(len(train_iter.dataset))
# print(len(test_iter.dataset))

# 构造网络
net = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(320, 80),
    nn.ReLU(),
    nn.Linear(80, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
)

# 设置损失函数
loss = nn.CrossEntropyLoss()

# 设置优化器
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

#训练
MyModel.train(train_iter, test_iter, net, loss, optimizer, num_epochs,train_size)