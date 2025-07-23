import torch
from torchsummary import summary
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷积层和池化层部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, padding=2, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # 定义全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        # 先通过卷积层和池化层
        x = self.conv_layers(x)
        # 再通过全连接层
        x = self.fc_layers(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))