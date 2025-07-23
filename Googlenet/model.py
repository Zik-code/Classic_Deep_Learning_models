import torch
from torch import nn
from torchsummary import summary

# 搭建inception
class Inception(nn.Module):
    # 将通道数定义为参数，一个路径包含连个卷积的定义为元组
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.ReLU = nn.ReLU()
        # 路径1
        self.p1_1=  nn.Conv2d(in_channels = in_channels  ,out_channels = c1 ,kernel_size= 1 )
        # 路径2
        self.p2_1=  nn.Conv2d(in_channels = in_channels ,out_channels = c2[0] ,kernel_size= 1 )
        self.p2_2=  nn.Conv2d(in_channels = c2[0] ,out_channels = c2[1] ,kernel_size= 3, padding = 1)
        # 路径3
        self.p3_1 = nn.Conv2d(in_channels = in_channels , out_channels= c3[0],kernel_size = 1)
        self.p3_2 = nn.Conv2d(in_channels = c3[0] , out_channels=c3[1],kernel_size= 5 ,padding= 2)
        # 路径4
        self.p4_1 = nn.MaxPool2d(kernel_size = 3,padding = 1,stride=1)
        self.p4_2 = nn.Conv2d(in_channels = in_channels , out_channels=c4,kernel_size = 1)
    # 单条路径里嵌套来写，每条路径输出结果单独用变量保存，以便最后融合
    def forward(self, x):
        p1 = self.ReLU(self.p1_1(x))
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))
        #print(p1.shape,p2.shape,p3.shape,p4.shape)
        # 返回通道融合的结果
        return torch.cat((p1,p2,p3,p4),dim=1)

class GoogLeNet(nn.Module):
    def __init__(self,Inception):
        super(GoogLeNet,self).__init__()
        # 省略原论文中的局部归一化操作
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels= 64,kernel_size = 7,stride = 2,padding = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1 )
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels = 192, kernel_size=3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )


        self.b3 = nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            # 上一层的通道合并后的输出为下一层的通道输入 64+128+32+32 = 256
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.b4 = nn.Sequential(
            # 128+192+96+64 = 480
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128,256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128),128),
            # 上一层的通道合并后的输出为下一层的通道输入 64+128+32+32 = 256
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,10)
        )
        # 参数初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode='fan_out',nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(Inception).to(device)
    print(summary(model,(1,224,224)))