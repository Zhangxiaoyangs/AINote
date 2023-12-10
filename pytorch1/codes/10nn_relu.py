#P20 非线性激活
#非线性激活就是为了给神经网络引入一些非线性特征，
#因为你非线性越多就更容易训练出符合各种曲线或者各种特征的模型
#最常见的 ReLU , Sigmoid
import torch
from torch import nn
from torch.nn import ReLU,Sigmoid
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

input = torch.tensor([[1,0.5],
                      [-1,3]])

input = torch.reshape(input=input,shape=(-1,1,2,2))
print(input.shape)
print(input)

class Austin(nn.Module):
    def __init__(self):
        super(Austin,self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output    

austin = Austin()

step = 0  
writer = SummaryWriter("logs")
for data in dataloader:
    images,tagets = data
    writer.add_images("ReLU input",images,step)
    output = austin(images)
    writer.add_images("ReLU output",output,step)
    step+=1

writer.close
