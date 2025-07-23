import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet

# 数据加载函数
def test_model_process():
    transform = transforms.Compose(
        [transforms.Resize(size=224),transforms.ToTensor()]
    )
    test_data =  FashionMNIST(root='./dataset',train = False,transform=transform,download=True)
    test_dataloader = Data.DataLoader(test_data,batch_size=1,shuffle=True,num_workers=0)
    return test_dataloader

def train_model_process(model,test_dataloader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 初始化参数,测试集这里只需要算正确率
    test_acc = 0.0
    test_num = 0

    with torch.no_grad():
        for imgs , labels in test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            model.eval()
            output = model(imgs)
            pre_lab = torch.argmax(output, dim=1)
            test_acc += torch.sum(pre_lab == labels.detach())

            test_num += imgs.size(0) # len(test_data)

    test_acc = test_acc.double().item() / test_num
    print('测试准确率为：',test_acc)


# 模型测试

if __name__ == '__main__':
    # 加载模型
    model = AlexNet()
    test_dataloader = model.load_state_dict(torch.load('best_model.pth'))
    test_model_process(model,test_dataloader)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 输出每一次的预测和真实值的类别
    classes = ['Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for imgs , labels in test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            # 设置模型为验证模式
            model.eval()
            output = model(imgs)
            pre_lab  = torch.argmax(output,dim=1)
           # 取出数值
            result = pre_lab.item()
            label = labels.item()

            print('预测值：',classes[result],'---------','真实值：',classes[label])
