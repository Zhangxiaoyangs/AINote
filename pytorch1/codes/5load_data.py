#P14 如何使用数据集
import torchvision
from torch.utils.tensorboard import SummaryWriter
#terminal tensorboard --logdir=logs  

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True)


# print(test_set[0])
# print(test_set.classes)
# img,target = test_set[0]
# print(target)
# print(test_set.classes[target])

# img.show()

writers = SummaryWriter("p10")
for i in range(10):
    img,target = test_set[i]
    dataset_tensor= dataset_transform(img=img)
    writers.add_image("test_set",dataset_tensor,i)


writers.close()   
