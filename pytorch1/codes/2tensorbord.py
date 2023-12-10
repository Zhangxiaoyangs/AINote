from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


### tensorboard 使用

writer = SummaryWriter("logs")
img_path = "pytorch1/dataset/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
writer.add_image("test",img_array,1,dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=x",i,i*2)

writer.close()

# 在 Termial 运行 tensorboard --logdir=logs  可以打开里面的图 