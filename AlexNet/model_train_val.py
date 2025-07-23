import copy
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from model import  AlexNet
from torchvision.datasets import FashionMNIST
import numpy as np
from tqdm import tqdm  # 导入 tqdm

# 数据加载函数
def train_val_train_process():
    transform = transforms.Compose(
        [transforms.Resize(size=224), transforms.ToTensor()]
    )
    dataset = FashionMNIST(root='./dataset', train=True,
                           transform=transform, download=True)
    # 划分
    train_data, val_data = Data.random_split(
        dataset, [round(0.8*len(dataset)), round(0.2*len(dataset))])
    train_dataloader = Data.DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=0)
    # 返回一个批次的数据元组，包含输入数据（图像）和对应的标签。
    val_dataloader = Data.DataLoader(
        val_data, batch_size=32, shuffle=False, num_workers=0)
    print(len(train_dataloader))
    print(len(val_dataloader))
    return train_dataloader, val_dataloader

# 训练和验证函数
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    start = time.time()

    # 使用 tqdm 包装最外层的训练循环
    for epoch in tqdm(range(num_epochs), desc='Training', unit='epoch'):
        print(f'-------第{epoch+1}次训练-------')
        train_loss = 0.0
        train_acc = 0
        val_loss = 0.0
        val_acc = 0
        train_num = 0
        val_num = 0

        # 训练阶段
        for step, (imgs, labels) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            model.train()
            # 输入一个批次图像到模型里
            output = model(imgs)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item()--该批次的平均损失 ， imgs.size(0)--该批次的图像数量 ， 该批次的总损失--train_loss
            train_loss += loss.item() * imgs.size(0)
            train_acc += torch.sum(pre_lab == labels.detach())
            # 取下个批次
            train_num += imgs.size(0)

        # 验证阶段，在验证集上不进行梯度更新和反向传播，否则模型就会根据验证集的数据进行调整，
        # 这样验证集就不再是独立于训练过程的数据，也就无法真实地反映模型的泛化能力了。
        for step, (imgs, labels) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            model.eval()
            output = model(imgs)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, labels)

            val_loss += loss.item() * imgs.size(0)
            val_acc += torch.sum(pre_lab == labels.data)
            val_num += imgs.size(0)

        # 计算平均损失和准确率
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc_all.append(train_acc.double().item() / train_num)
        val_acc_all.append(val_acc.double().item() / val_num)

        print('{} train loss:{:.4f} train acc: {:.4f}'.format(
            epoch, train_loss_all[epoch], train_acc_all[epoch]))
        print('{} val loss:{:.4f} val acc: {:.4f}'.format(
            epoch, val_loss_all[epoch], val_acc_all[epoch]))

        # 保存最佳模型
        if val_acc_all[epoch] > best_acc:
            best_acc = val_acc_all[epoch]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算耗时
        time_use = time.time() - start
        print('该轮次训练和验证耗费时间{:.0f}m{:.0f}s'.format(
            time_use // 60, time_use % 60))

    # 保存最佳模型
    torch.save(best_model_wts, './VGG16/best_model.pth')

    # 返回训练过程数据
    train_process = pd.DataFrame(data={'epoch': range(num_epochs),
                                       'train_loss_all': train_loss_all,
                                       'val_loss_all': val_loss_all,
                                       'train_acc_all': train_acc_all,
                                       'val_acc_all': val_acc_all})
    return train_process

import matplotlib.pyplot as plt

# 绘图函数
def matplot_acc_loss(train_process):
    # 创建一个新的图形窗口，并设置图形的大小为宽度 12 英寸，高度 4 英寸
    plt.figure(figsize=(12, 4))

    # 在图形窗口中创建一个 1 行 2 列的子图布局，并指定当前操作的是第 1 个子图
    # 这个子图用于绘制训练集和验证集的损失曲线
    plt.subplot(1, 2, 1)

    # 绘制训练集的损失曲线
    # train_process['epoch'] 作为 x 轴数据，表示训练的轮次
    # train_process.train_loss_all 作为 y 轴数据，表示每一轮训练集的损失值
    # 'ro-' 是线条和标记的格式字符串，'r' 表示红色，'o' 表示圆形标记，'-' 表示实线连接标记
    # label='train loss' 为该曲线设置一个标签，用于在图例中显示
    plt.plot(train_process['epoch'], train_process.train_loss_all,
             'ro-', label='train loss')

    plt.plot(train_process['epoch'], train_process.val_loss_all,
             'bs-', label='val loss')

    # 显示图例，通过图例可以清晰地分辨出哪条曲线代表训练集损失，哪条曲线代表验证集损失
    plt.legend()

    # 为 x 轴添加标签，表明 x 轴表示训练轮次
    plt.xlabel('epoch')
    # 为 y 轴添加标签，表明 y 轴表示损失值
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all,
             'ro-', label='train acc')

    plt.plot(train_process['epoch'], train_process.val_acc_all,
             'bs-', label='val acc')

    # 显示图例，通过图例可以清晰地分辨出哪条曲线代表训练集准确率，哪条曲线代表验证集准确率
    plt.legend()

    # 为 x 轴添加标签，表明 x 轴表示训练轮次
    plt.xlabel('epoch')
    # 为 y 轴添加标签，表明 y 轴表示准确率
    plt.ylabel('acc')

    # 显示绘制好的图形，将包含损失曲线和准确率曲线的图形窗口展示出来
    plt.show()

# 主函数
if __name__ == '__main__':
    alexnet = AlexNet()
    train_dataloader, val_dataloader = train_val_train_process()
    train_process = train_model_process(
        alexnet, train_dataloader, val_dataloader, num_epochs=20)
    matplot_acc_loss(train_process)