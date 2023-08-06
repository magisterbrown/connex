import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.masked import as_masked_tensor
import numpy as np

class ConnNet(nn.Module):
    def __init__(self,  cols: int, rows: int, mark=1):
        super().__init__()
        self.cols=cols
        self.rows=rows
        self.mark=mark
        self.dtype=torch.float32
        inside = 128

        self.l1 = nn.Conv2d(2,inside,3,padding=1)
        self.bn = nn.BatchNorm2d(inside)
        res_blocks = [BasicBlock(inside, inside)]*5
        self.body = nn.ModuleList(res_blocks)

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

        return x

    def policy_head(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.01)
        x = self.bn2(x)
        x = x.view(x.shape[0],-1) 
        x = self.fc(x)

        return x

    def convert(self, x: list) -> torch.tensor:
        field = np.array(x)
        field = np.resize(x,(self.rows, self.cols))
        tn1 = torch.tensor(field==self.mark,dtype=self.dtype)
        others = np.logical_and(field!=self.mark,field!=0)
        tn2 = torch.tensor(others,dtype=self.dtype)
        return torch.stack((tn1,tn2)).unsqueeze(0)

    def forward(self, x):
        x = self.convert(x)
        x = self.l1(x)
        x = F.leaky_relu(x,0.01)
        x = self.bn(x)
        
        value = self.value_head(x)
        policy = self.policy_head(x)
        return policy, value

    def play(self, x):
        policy, _ = self.forward(x)
        return self.get_turn(x, policy) 

    def get_turn(self, field: list, policy: torch.Tensor) -> int:
        tops = torch.tensor(field[:self.cols])==0
        mat = as_masked_tensor(policy[0], tops)
        mxi = torch.masked.argmax(mat)
        return int(mxi.item())

from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



