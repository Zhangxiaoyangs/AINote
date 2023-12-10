from torch.utils.data import Dataset
from PIL import Image
import os
### 数据集获取
class MyData(Dataset):

    def __init__(self,root_dir,lable_dir) -> None:
        self.root_dir=root_dir
        self.lable_dir=lable_dir
        self.path = os.path.join(self.root_dir,self.lable_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.lable_dir,img_name)
        img = Image.open(img_item_path)
        label =  self.lable_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "pytorch1/dataset/train"
ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
# img , lable = ants_dataset[1]
# img2 , lable2 = bees_dataset[1]

# print("ants "+img.show()+"  =  "+lable)
# print("bees "+img2.show()+"  =  "+lable2)


print(ants_dataset.__len__())


