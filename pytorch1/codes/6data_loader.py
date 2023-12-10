#P15 dataLoader 使用
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader 
#terminal tensorboard --logdir=logs  

test_data = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=torchvision.transforms.ToTensor())
#DataLoader 类似一个对数据加载的加载器，加载什么数据？ 如何加载 就使用DataLoader
test_loader  = DataLoader(dataset=test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=False)

#测试数据集中的第一张图片以及Target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs,targets = data
    writer.add_images("test_data",imgs, step)
    step = step + 1

writer.close()
print("Over")
