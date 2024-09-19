import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from garmentclassifier import GarmentClassifier

def get_latest_model_path(directory, pattern="model_*.pth"):
    # 获取目录下所有符合模式的文件
    model_files = glob.glob(os.path.join(directory, pattern))
    if not model_files:
        raise FileNotFoundError("No model files found in the directory.")
    
    # 找到最新的模型文件
    latest_model_file = max(model_files, key=os.path.getmtime)
    return latest_model_file


# 定义图像转换操作：将图像转换为张量，并进行归一化处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小为28x28
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练好的模型
model = GarmentClassifier()
model_path = get_latest_model_path('./')  # 获取最新的模型文件
model.load_state_dict(torch.load(model_path, weights_only=False)) # 加载模型参数
model.eval()  # 设置模型为评估模式

# 从本地加载图像
image_path = 'shoe.jpg'  # 替换为实际的图像路径
image = Image.open(image_path).convert('L')  # 将图像转换为灰度图

# 预处理图像
image = transform(image)
image = image.unsqueeze(0)  # 增加一个批次维度

# 推理（预测）
with torch.no_grad():  # 在推理过程中不需要计算梯度
    outputs = model(image)  # 前向传播，计算模型输出
    _, predicted = torch.max(outputs, 1)  # 获取预测结果

# 定义类别名称
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# 打印预测结果
print(f'Predicted label: {classes[predicted.item()]}')