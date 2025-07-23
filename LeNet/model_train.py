import os

from mpmath.identification import transforms

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
from torchvision import transforms

def train_val_data_process():
    transform = transforms.Compose([transforms.Resize(size = 28)],transforms.Totensor())
    dataset = FashionMNIST(root='./dataset',train=True,transform=transform)

    train_data,val_data = Data.random_split(dataset,[round(0.8*len(dataset),round(0.2*len(dataset)))])

    train_dataloader = DataLoader(train_data,batch_size=64,shuffle=True,num_workers=8)
    val_dataloader = DataLoader(val_data,batch_size=64,shuffle=True,num_workers=8)

    return train_dataloader,val_dataloader