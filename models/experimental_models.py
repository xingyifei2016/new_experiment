import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import layers
import math
from pdb import set_trace as st

def m(x):
    return True in torch.isnan(x).cpu().detach().numpy()


class Previous(nn.Module):
    def __init__(self):
        super(Previous, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransformUpsample(20, (22, 22), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 30, (5, 5), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(30, 40, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(40, 50, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.proj2(x1)
        x3 = self.complex_conv2(x2)
        x4 = self.proj2(x3)
        x5 = self.linear_1(x4)
        x6 = self.conv_1(x5)
        x = self.bn_1(x6)
        x_res = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
class MoreLayers(nn.Module):
    def __init__(self):
        super(MoreLayers, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv3 = layers.ComplexConv(20, 20, (4, 4), (1, 1), num_tied_block=1)
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)

        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(20, 30, (5, 5), (3, 3))
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 40, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(40)
        self.linear_2 = nn.Linear(40, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.proj2(x1)
        x3 = self.complex_conv2(x2)
        x4 = self.proj2(x3)
        x4 = self.complex_conv3(x4)
        x4 = self.proj2(x4)
        x5 = self.linear_1(x4)
       
        x = self.mp_1(x5)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
class LessLayers(nn.Module):
    def __init__(self):
        super(LessLayers, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = nn.Conv2d(20, 30, (5, 5), (2, 2))
        self.complex_conv3 = nn.Conv2d(30, 40, (4, 4), (1, 1))
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)

        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3))
        self.bn_2 = nn.BatchNorm2d(50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(50, 60, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(60)
        self.linear_2 = nn.Linear(60, 40)
        self.linear_4 = nn.Linear(40, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.proj2(x1)
        x = self.linear_1(x2)
        x = self.complex_conv2(x)
        x=self.relu(x)
        x4 = self.complex_conv3(x)
        x4 = self.relu(x4)
        x = self.mp_1(x4)
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
    
class Dis_Tan3(nn.Module):
    def __init__(self):
        super(Dis_Tan3, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 10, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(10, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv3 = layers.ComplexConv(20, 30, (5, 5), (2, 2), num_tied_block=1)
        
        self.distance1 = layers.DifferenceLayer(10, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        self.distance3 = layers.DifferenceLayer(30, (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(30, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(30, 40, (3, 3), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (2, 2), (1, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
        x_ = self.tangentReLU(x2)
        x3 = self.complex_conv2(x_)
        x__ = self.distance2(x3)
        x___ = self.tangentReLU(x__)
        x3 = self.complex_conv3(x___)
        x__ = self.distance3(x3)
        x___ = self.tangentReLU(x__)
        x5 = self.linear_1(x___)
        x6 = self.conv_1(x5)
        x = self.bn_1(x6)
        x_res = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
class PreviousDisTan(nn.Module):
    def __init__(self):
        super(PreviousDisTan, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        self.distance1 = layers.DifferenceLayerUpsample(20, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayerUpsample(20, (2, 2), num_tied_block=1)
        
        
        self.tangentReLU = layers.tangentRELU()
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransformUpsample(20, (22, 22), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 30, (5, 5), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(30, 40, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(40, 50, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 30)
        self.linear_4 = nn.Linear(30, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
        x2 = self.tangentReLU(x2)
        x3 = self.complex_conv2(x2)
        x2 = self.distance2(x3)
        x2 = self.tangentReLU(x2)
        x5 = self.linear_1(x2)
        x6 = self.conv_1(x5)
        x = self.bn_1(x6)
        x_res = self.relu(x)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
class LessLayersDis(nn.Module):
    def __init__(self):
        super(LessLayersDis, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = nn.Conv2d(20, 30, (5, 5), (2, 2))
        self.complex_conv3 = nn.Conv2d(30, 40, (4, 4), (1, 1))
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.distance1 = layers.DifferenceLayerUpsample(20, (2, 2), num_tied_block=1)
        
        
        self.tangentReLU = layers.tangentRELU()
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3))
        self.bn_2 = nn.BatchNorm2d(50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(50, 60, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(60)
        self.linear_2 = nn.Linear(60, 40)
        self.linear_4 = nn.Linear(40, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
        x2 = self.tangentReLU(x2)
        x = self.linear_1(x2)
        x = self.complex_conv2(x)
        x=self.relu(x)
        x4 = self.complex_conv3(x)
        x4 = self.relu(x4)
        x = self.mp_1(x4)
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
    
class MoreLayersDisTan(nn.Module):
    def __init__(self):
        super(MoreLayersDisTan, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 10, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(10, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv3 = layers.ComplexConv(20, 30, (4, 4), (1, 1), num_tied_block=1)
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(30, (2, 2), num_tied_block=1)
        
        self.distance1 = layers.DifferenceLayerUpsample(10, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayerUpsample(20, (2, 2), num_tied_block=1)
        self.distance3 = layers.DifferenceLayerUpsample(30, (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(30, 40, (5, 5), (3, 3))
        self.bn_2 = nn.BatchNorm2d(40)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(40, 50, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(50)
        self.linear_2 = nn.Linear(50, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
        x2 = self.tangentReLU(x2)
        x3 = self.complex_conv2(x2)
        x2 = self.distance2(x3)
        x4 = self.tangentReLU(x2)
        x4 = self.complex_conv3(x4)
        x2 = self.distance3(x4)
        x2 = self.tangentReLU(x2)
        x5 = self.linear_1(x2)
        
        x = self.mp_1(x5)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x