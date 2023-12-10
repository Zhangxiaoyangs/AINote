#P16 使用神经网络  torch.nn
import torch
import torch.nn as nn
# 定义类
class Austin(nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()
        self.a = torch.tensor(2.0)

    # 反向传播    
    def forward(self,imput):
        output = imput + self.a 
        return output
        
austin = Austin()
x = torch.tensor(1.0)
output = austin(x)
print(output)
