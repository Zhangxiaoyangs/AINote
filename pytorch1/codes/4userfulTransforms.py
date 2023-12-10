#常用的transforms 方法 包括 Noralize
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#terminal tensorboard --logdir=logs  
writer = SummaryWriter("logs")
img_path = "pytorch1/dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

#Totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor",img_tensor)

#Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,.5,0.5])
imag_norm = trans_norm(img_tensor)
print(imag_norm[0][0][0])
writer.add_image("Normalize",imag_norm,2)

#Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
#PIL -> resize -> iamge_resize PIL
img_resize = trans_resize(img)
# iamge_resize PIL -> totensor -> img_resized toensor
img_resize  = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(type(img_resize))

#Compose - resize -2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor 
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resized_2 = trans_compose(img)
writer.add_image("Compose Resize",img_resized_2,0)

#RandomCrop
trans_random = transforms.RandomCrop(100)
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop ",img_crop,i)

writer.close()
