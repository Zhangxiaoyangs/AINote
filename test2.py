import torch
from PIL import Image
import os
x = torch.rand(5, 3)
print(x)


t = torch.cuda.is_available()
print(t)

device = torch.device('cuda') if  torch.cuda.is_available() else torch.device('cpu')
print(device)

print('version '+ torch.__version__)

img_path = "C:\\Users\\PC\Desktop\\pyTF\\pytorch1\\dataset\\train\\ants\\0013035.jpg"

img = Image.open(img_path)
print(img)

print(img.size)
# img.show()

dir_path = "/dataset/train/ants"
img_path_list = os.listdir(dir_path)
print(img_path_list)

