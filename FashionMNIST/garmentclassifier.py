import torch.nn as nn
import torch.nn.functional as F

# 定义一个用于服装分类的卷积神经网络
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        # 定义第一个卷积层，输入通道数为1，输出通道数为6，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 定义最大池化层，池化窗口大小为2x2
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一个全连接层，输入大小为16*4*4，输出大小为120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 定义第二个全连接层，输入大小为120，输出大小为84
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层，输入大小为84，输出大小为10（对应10个类别）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 通过第一个卷积层和ReLU激活函数，然后通过最大池化层
        x = self.pool(F.relu(self.conv1(x)))
        # 通过第二个卷积层和ReLU激活函数，然后通过最大池化层
        x = self.pool(F.relu(self.conv2(x)))
        # 展平张量，从多维张量变为二维张量
        x = x.view(-1, 16 * 4 * 4)
        # 通过第一个全连接层和ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层和ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三个全连接层（输出层）
        x = self.fc3(x)
        return x
    