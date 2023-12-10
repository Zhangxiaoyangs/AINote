#P17 P18 卷积层
import torch.nn.functional as F
import torch
from torch import  nn
from torch.nn import Conv2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#terminal tensorboard --logdir=logs  

#-----------------------------P17-------------------------------------
#数据模拟conv2d的计算过程演示
#数据
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
# 卷积核
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])
# Pytorch 需要有四个数字 但是这里只有2个数字
print(input.shape)
print(kernel.shape)

# 使用尺寸变换来重新定义形状
input  = torch.reshape(input= input,shape= (1,1,5,5))
kernel  = torch.reshape(input= kernel,shape= (1,1,3,3))
print(input.shape)
print(kernel.shape)

output  = F.conv2d(input,kernel,padding=0,stride=2)
print(output)


#-----------------------------P18-------------------------------------
#卷积Git文档：  https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

class Austin(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

austin = Austin()
print(austin)     
step = 0  
writer = SummaryWriter("logs")
for data in dataloader:
    img , targets = data
    output = austin(img)
    print(img.shape)
    print(output.shape)
    #torch.Size([64, 3, 32, 32])
    writer.add_images("input",img,step)
    #torch.Size([64, 6, 30, 30])
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step+=1

writer.close

#vgg-16论文   https://www.geeksforgeeks.org/vgg-16-cnn-model/