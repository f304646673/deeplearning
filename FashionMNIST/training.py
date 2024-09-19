from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from garmentclassifier import GarmentClassifier

# 定义图像转换操作：将图像转换为张量，并进行归一化处理
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]) # 对图像的每个通道进行标准化，使得每个通道的像素值具有零均值和单位标准差

# 加载FashionMNIST训练数据集，并应用定义的图像转换操作
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform)

# 创建数据加载器，用于批量加载训练数据，batch_size为4，数据顺序随机打乱
trainloader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)

# 实例化模型
model = GarmentClassifier()
# 定义损失函数为交叉熵损失
loss_fn = torch.nn.CrossEntropyLoss()
# 定义优化器为随机梯度下降（SGD），学习率为0.001，动量为0.9
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型，训练2个epoch
for epoch in range(2):
    running_loss = 0.0  # 初始化累计损失
    # 枚举数据加载器中的数据，i是批次索引，data是当前批次的数据
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data  # 获取输入数据和对应的标签
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播，计算模型输出
        loss = loss_fn(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累加损失

        # 每2000个批次打印一次平均损失
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')
            running_loss = 0.0  # 重置累计损失
      
# 获取当前时间戳，格式为 'YYYYMMDD_HHMMSS'
timestamp = datetime.now().strftime('%Y%m%d%H%M%S.pth')

# 定义模型保存路径，包含时间戳
model_path = 'model_{}'.format(timestamp)      

# 保存模型的状态字典到指定路径
torch.save(model.state_dict(), model_path)