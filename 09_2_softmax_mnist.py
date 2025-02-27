# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 64

# MNIST Dataset 手写数字识别
# datasets.MNIST是下载MNIST这个数据集
train_dataset = datasets.MNIST(root='./mnist_data/',  # 根目录
                               train=True,  # 加载训练集，如果为False为加载测试集
                               transform=transforms.ToTensor(),  # 一个预处理操作，将 PIL 图像或 NumPy 数组转换为浮点数张量，并将像素值归一化到 [0, 1]
                               download=True)  # 是否下载数据集

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,  # 下载测试集
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)  # 每个epoch打乱顺序

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)  # 每个epoch打乱顺序


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# momentum是动量系数，能帮助逃离局部最小值，会优化梯度计算公式，在原有基础上再减去一个动量，动量=上一次动量乘以动量系数加上学习率乘以梯度
# SGD实现随机梯度下降
def train(epoch):
    model.train()  # 将模型设置为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):  # enumerate，拿到每一个元素及其索引
        data, target = torch.tensor(data), torch.tensor(target)
        optimizer.zero_grad()  # 清零梯度
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()  # 模型改成推理模式
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data, target
        output = model(data)
        # sum up batch loss
        # 计算损失总和
        test_loss += criterion(output, target).item()

        # get the index of the max
        pred = output.max(1, keepdim=True)[1]  # 这里是一个概率，所以要获取预测的最大的概率，即最有可能的数字
        correct += pred.eq(target.view_as(pred)).cpu().sum()
        #pred是模型预测的类别索引，形状为[batch_size,1]，target.view_as(pred)是把预测结果转化为和pred形状相同
        #eq是判断等不等，返回一个bool值，.cpu()是移到CPU上，因为GPU算的不快，.sum()是求和

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),#计算平均损失
        100. * correct / len(test_loader.dataset)))#计算准确率，格式化为百分比。


for epoch in range(1, 10):
    train(epoch)
    test()
