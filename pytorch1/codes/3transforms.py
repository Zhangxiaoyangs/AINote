from torchvision import transforms as tf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

###
# 
# transfrom 主要作用是对图片进行一些变化
# python的用法 -> tensor数据类型
# 通过 tf.ToTensor去看2个问题
# 1 transforms 如何使用
# 2 为什么我们需要Tensor数据类型？

img_path = "pytorch1/dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")

# 1 transforms 如何使用
tensor_transform =  tf.ToTensor()
tensor_img = tensor_transform(img)
print(tensor_img)

writer.add_image("Tenosr_img",tensor_img)
writer.close()


