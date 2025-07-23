import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from model import LeNet
import torch.utils.data as Data

# 定义数据预处理转换
transform = torchvision.transforms.Compose([
    # 将图像大小调整为 224x224
    torchvision.transforms.Resize((224, 224)),
    # 将图像转换为张量
    torchvision.transforms.ToTensor()
])

# 加载训练集
train_dataset = FashionMNIST(root='./dataset', transform=transform, train=True, download=True)
# 加载测试集
test_dataset = FashionMNIST(root='./dataset', transform=transform, train=False, download=True)

# 创建训练集数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
# 创建测试集数据加载器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 遍历训练集数据加载器，只取一个批次的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
# 将四维张量移除第 1 维（通道维度），并转换成 Numpy 数组
batch_x = b_x.squeeze().numpy()
# 将张量转换成 Numpy 数组
batch_y = b_y.numpy()
# 获取数据集的类别标签
class_label = train_dataset.classes
with open('class_labels.json', 'w') as f:
    json.dump(class_label, f)


# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()