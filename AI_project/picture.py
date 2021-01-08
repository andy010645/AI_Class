import numpy as np
import torch
import torch.nn as nn
import torch,torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import sys
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.datasets import ImageFolder
from torch import optim
from torchsummary import summary
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from food_train import BuildModel

writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

test_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])      

def insert_image():
    C = BuildModel() # from food_train.py
    C = torch.load('resnet50_epoch30.pt',map_location=torch.device('cpu')) # load model
    ins = sys.argv[1]
    print("Picture "+str(ins),"\n")
    image = Image.open('test\\'+str(ins)+'.jpg').convert('RGB')
    plt.imshow(image) # show select picture

    t = test_transformer
    image = t(image)
    image = torch.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
    predict_class = C.forward(image)
    p = F.softmax(predict_class,dim=1) # 各 label 機率
    p = p.detach().numpy() # tensor to numpy
    fp = open('food\\meta\\meta\\classes.txt','r')
    label = fp.read().splitlines() # 以'\n'切資料\
    '''
    plt.figure(figsize=(10,10))    # 顯示圖框架大小
    plt.pie(p[0].tolist(), # numpy to list
            labels = label,                # label
            autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
            #pctdistance = 0.6,              # 數字距圓心的距離
            textprops = {"fontsize" : 10},  # 文字大小
            shadow=False)                    # 設定陰影
   
    plt.axis('equal')                                          # 使圓餅圖比例相等
    #plt.legend(loc = "best")                                   # 設定圖例及其位置為最佳
'''
    for i in range(101):
        if p[0][i] > 0.02:
            print(label[i])
            print(p[0][i])
            print("\n")
    plt.show()

def show_image():
    t = test_transformer
    image = Image.open('test\\823084.jpg').convert('RGB')
    plt.imshow(image)
    plt.show()
if __name__ == '__main__':

    insert_image()


    