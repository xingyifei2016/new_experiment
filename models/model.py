import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import layers
import math
from pdb import set_trace as st
import shrinkage_layers

    
    
class ManifoldNetRes(nn.Module):
    def __init__(self):
        super(ManifoldNetRes, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
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
    
class Dis_Tan(nn.Module):
    def __init__(self):
        super(Dis_Tan, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        
        
        self.distance1 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
            
        x_ = self.tangentReLU(x2)
            
        x3 = self.complex_conv2(x_)
            
        x__ = self.distance2(x3)
            
        x___ = self.tangentReLU(x__)
           
            
        x5 = self.linear_1(x___)
        
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
    
    
class Tan_Dis(nn.Module):
    def __init__(self):
        super(Tan_Dis, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        
        
        self.distance1 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        
        x1 = self.tangentReLU(x1)
        
        
        x2 = self.distance1(x1)
            
        
            
        x3 = self.complex_conv2(x2)
        
        
        x3 = self.tangentReLU(x3)
            
        x__ = self.distance2(x3)
            
           
            
        x5 = self.linear_1(x__)
        
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
    
    
class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        
        
        self.distance1 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        
        
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
        x3 = self.complex_conv2(x2)
        x__ = self.distance2(x3)
        x5 = self.linear_1(x__)
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
    
class NL(nn.Module):
    def __init__(self):
        super(NL, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        x2 = self.tangentReLU(x1)
        x3 = self.complex_conv2(x2)
        x__ = self.tangentReLU(x3)
        x5 = self.linear_1(x__)
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
    
class RealModel(nn.Module):
    def __init__(self):
        super(RealModel, self).__init__()
        self.complex_conv1 = nn.Conv2d(3, 20, (5, 5), (2, 2))
        self.complex_conv2 = nn.Conv2d(20, 20, (5, 5), (2, 2))
        
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        x = x.squeeze(2)
        x1 = self.complex_conv1(x)
        
        x_ = self.relu(x1)
            
        x3 = self.complex_conv2(x_)
        
        x5 = self.relu(x3)
           
        
        
        x6 = self.conv_1(x5)
        
        x = self.bn_1(x6)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x_res = self.relu(x)
        
#         x = self.id1(x_res) + self.res1(x_res)
        
        x = self.mp_1(x)
        x = self.conv_2(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.bn_2(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x_res = self.relu(x)
#         x = self.id2(x_res) + self.res2(x_res)
        
        x = self.conv_3(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.bn_3(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
    
class ManifoldNetRes3(nn.Module):
    def __init__(self):
        super(ManifoldNetRes3, self).__init__()
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
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
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
    
class Dis_Tan3(nn.Module):
    def __init__(self):
        super(Dis_Tan3, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 10, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(10, 15, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv3 = layers.ComplexConv(15, 20, (5, 5), (2, 2), num_tied_block=1)
        
        self.distance1 = layers.DifferenceLayer(10, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(15, (2, 2), num_tied_block=1)
        self.distance3 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
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
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
    
class Dis3(nn.Module):
    def __init__(self):
        super(Dis3, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        
        
        self.distance1 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        self.distance2 = layers.DifferenceLayer(20, (2, 2), num_tied_block=1)
        
        
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        x2 = self.distance1(x1)
        x3 = self.complex_conv2(x2)
        x__ = self.distance2(x3)
        x5 = self.linear_1(x__)
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
    
class NL3(nn.Module):
    def __init__(self):
        super(NL3, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        
        self.tangentReLU = layers.tangentRELU()
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        
        x1 = self.complex_conv1(x)
        x2 = self.tangentReLU(x1)
        x3 = self.complex_conv2(x2)
        x__ = self.tangentReLU(x3)
        x5 = self.linear_1(x__)
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
    
class RealModel(nn.Module):
    def __init__(self):
        super(RealModel, self).__init__()
        self.complex_conv1 = nn.Conv2d(3, 20, (5, 5), (2, 2))
        self.complex_conv2 = nn.Conv2d(20, 20, (5, 5), (2, 2))
        
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(20, 25, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(25, 30, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(25)
        self.bn_2 = nn.BatchNorm2d(30)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(30, 35, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(35)
        self.linear_2 = nn.Linear(35, 20)
        self.linear_4 = nn.Linear(20, 10)
        
       
        
    def forward(self, x):
        x = x.squeeze(2)
        x1 = self.complex_conv1(x)
        
        x_ = self.relu(x1)
            
        x3 = self.complex_conv2(x_)
        
        x5 = self.relu(x3)
           
        
        
        x6 = self.conv_1(x5)
        
        x = self.bn_1(x6)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x_res = self.relu(x)
        
#         x = self.id1(x_res) + self.res1(x_res)
        
        x = self.mp_1(x)
        x = self.conv_2(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.bn_2(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x_res = self.relu(x)
#         x = self.id2(x_res) + self.res2(x_res)
        
        x = self.conv_3(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.bn_3(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x

def calc_next(inputs, kern, stride, outs):
    
    if type(kern) == int:
        dims = int(math.floor((inputs-(kern-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, dims])
    else:
        dims = int(math.floor((inputs-(kern[0]-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, 1])
    
    
class shrinkage(nn.Module):
    def __init__(self, num_distr, ws, partial=False):
        super(shrinkage, self).__init__()
        self.partial=partial
        self.complex_conv1 = shrinkage_layers.ComplexConv2Deffangle4Dxy(1, 20, (5, 5), (2, 2))
        self.proj1 =shrinkage_layers.ReLU4Dsp(20)
        self.complex_conv2 = shrinkage_layers.ComplexConv2Deffangle4Dxy(20, 20, (5, 5), (2, 2))
        params={'num_classes': 11, 'num_distr': num_distr}
        self.SURE = shrinkage_layers.SURE_pure(params, (2, 20, 22, 22), 20, ws = ws)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 30, (3, 3))
        self.mp_1 = nn.MaxPool2d((2,2))
        self.conv_2 = nn.Conv2d(30, 40, (3, 3))
        self.mp_2 = nn.MaxPool2d((3,3))
        self.conv_3 = nn.Conv2d(40, 60, (2, 2))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(40)
        self.bn_3 = nn.BatchNorm2d(60)
        self.linear_2 = nn.Linear(60, 30)
        self.linear_3 = nn.Linear(30, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Regular Network"
    def forward(self, x, labels=None):
        
        if self.partial:
            x = self.relu(x)
            x = self.conv_1(x)
            x = self.bn_1(x)
            x = self.relu(x)
            x = self.mp_1(x)
            x = self.conv_2(x)
            x = self.bn_2(x)
            x = self.relu(x)
            x = self.mp_2(x)
            x = self.conv_3(x)
            x = self.bn_3(x)
            x = self.relu(x)
            x = x.squeeze(-1).squeeze(-1)
            x = self.linear_2(x)
            x = self.relu(x)
            x = self.linear_3(x)
            
            return x
            
        else:
            
            x = self.complex_conv1(x)
            x = self.proj1(x)
            x = self.complex_conv2(x)
            x = self.proj1(x)
            x, losses, means = self.SURE(x, labels)
            
            means = means[:, 0, ...]* (self.SURE.weight[0]**2) + means[:, 1, ...]* (self.SURE.weight[1]**2)
            x = self.relu(x)
            x = self.conv_1(x)
            x = self.bn_1(x)
            x = self.relu(x)
            x = self.mp_1(x)
            x = self.conv_2(x)
            x = self.bn_2(x)
            x = self.relu(x)
            x = self.mp_2(x)
            x = self.conv_3(x)
            x = self.bn_3(x)
            x = self.relu(x)
            x = x.squeeze(-1).squeeze(-1)
            x = self.linear_2(x)
            x = self.relu(x)
            x = self.linear_3(x)
        
        
        
        
        
        res_loss = 0
        if losses is not None:
            res_loss = losses * (self.loss_weight ** 2)
        return x, res_loss, means
    

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class ResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=11, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
    
class ResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=11, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, input_channel=2):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
    
class ResNet3(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=11, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, input_channel=3):
        super(ResNet3, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
    
    
    
