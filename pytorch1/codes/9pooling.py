#P19 池化层
#最大池化的作用是什么？？？  最大池化是保留特征的同时把数据量减小
import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]],dtype=torch.float32)

input = torch.reshape(input=input,shape=(-1,1,5,5))
print(input.shape)


class Austin(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,input):
        x = self.maxpool1(input)
        return x

austin = Austin()
# output = austin(input)
#tensor([[[[2., 3.],[5., 1.]]]])   这个结果可以自己算下如何计算出来的
# print(output)    

step = 0  
writer = SummaryWriter("logs")

for data in dataloader:
    images,tagets = data
    writer.add_images("MaxPool input",images,step)
    output = austin(images)
    writer.add_images("MaxPool output",output,step)
    step+=1

writer.close
