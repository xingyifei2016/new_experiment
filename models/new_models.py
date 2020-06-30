import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from . import layers
import math
from pdb import set_trace as st

def z(inputs):
    return True in torch.isnan(inputs).cpu().detach().numpy()
    
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 10, (3, 3), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(10, 20, (3, 3), (2, 2), num_tied_block=1)
        
        self.distance1 = layers.DifferenceLayer(10, (3, 3), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(20, (3, 3), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        
        self.relu = nn.ReLU()
        
        self.linear_1 = layers.DistanceTransform(20, (3, 3), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 30, (3, 3), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(30, 40, (3, 3), (2, 2))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(40, 50, (3, 3), (1, 1))
        self.bn_3 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 10)
        
       
        
    def forward(self, x):
        x = self.complex_conv1(x)
     
        x = self.distance1(x)
        x = self.tangentReLU(x)
#         print(z(x))
        x = self.complex_conv2(x)
        x = self.distance2(x)
#         print(z(x))
        x = self.tangentReLU(x)
#         print(z(x))
        x = self.linear_1(x)
        
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x