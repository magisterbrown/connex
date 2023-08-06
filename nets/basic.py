import torch
import numpy.ma as ma
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
import numpy as np

class ConnNet(nn.Module):
    def __init__(self,  cols: int, rows: int):
        super().__init__()
        self.cols=cols
        self.rows=rows
        self.mark_map={1:1,2:0}
        self.dtype=torch.float32
        inside = 128

        self.l1 = nn.Conv2d(2,inside,3,padding=1)
        self.bn = nn.BatchNorm2d(inside)
        res_blocks = [BasicBlock(inside, inside) for i in range(5)]
        self.body = nn.Sequential(*res_blocks)

        #Policy head
        prepol = 32
        self.conv1 = nn.Conv2d(inside, prepol, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(prepol)
        self.fc = nn.Linear(cols*rows*prepol, self.cols)

        #Value head
        self.conv3 = nn.Conv2d(inside, 3, kernel_size=1) # value head
        self.bn3= nn.BatchNorm2d(3)
        self.fc3 = nn.Linear(3*rows*cols, 32)
        self.fc4 = nn.Linear(32, 1)

        self.to(dtype=self.dtype)

    def value_head(self, x):
        x = self.conv3(x)
        x = F.leaky_relu(x,0.01)
        x = self.bn3(x)
        x = x.view(x.shape[0],-1) 
        x = self.fc3(x)
        x = F.leaky_relu(x,0.01)
        x = self.fc4(x)
        x = torch.tanh(x)

        return x

    def policy_head(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.01)
        x = self.bn2(x)
        x = x.view(x.shape[0],-1) 
        x = self.fc(x).softmax(dim=1)

        return x

    def convert(self, x: list, mark=1) -> torch.tensor:
        field = np.array(x)
        field = np.resize(x,(self.rows, self.cols))
        tn1 = torch.tensor(field==mark,dtype=self.dtype)
        others = np.logical_and(field!=mark,field!=0)
        tn2 = torch.tensor(others,dtype=self.dtype)
        return torch.stack((tn1,tn2)).unsqueeze(0)

    def forward(self, x):
        board, player = x
        br = self.l1(board)
        br = self.bn(br)
        br = F.leaky_relu(br,0.01)
        br = self.body(br)
        
        value = self.value_head(br)
        policy = self.policy_head(br)
        return policy, value

    def play(self, x):
        return torch.argmax(self.get_probs(x)).item()

    def get_probs(self, x, mark=1):
        board = self.convert(x, mark)
        tensor_mark = torch.tensor([[mark]])
        policy, value = self.forward((board, tensor_mark))
        policy=policy[0]
        ilegal_moves = np.array(x[:self.cols])!=0
        policy[ilegal_moves] = 0
        return policy, value

