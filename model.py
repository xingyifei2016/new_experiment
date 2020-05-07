import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import layers
import math
from pdb import set_trace as st




class ManifoldNetRes(nn.Module):
    def __init__(self):
        super(ManifoldNetRes, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 5), (2, 2), num_tied_block=1)
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 5), (2, 2), num_tied_block=1)
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.linear_1 = layers.DistanceTransform(20, (2, 2), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 30, (4, 4), (1, 1))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2), (1, 1))
        self.bn_3 = nn.BatchNorm2d(70)
        self.linear_2 = nn.Linear(70, 30)
        self.linear_4 = nn.Linear(30, 11)
        self.res1=nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2=nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))
        
        
    def make_res_block(self, in_channel, out_channel):    
        res_block = []
        res_block.append(nn.BatchNorm2d(in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), out_channel, (1, 1), bias=False))
        return res_block
       
        #True in torch.isnan(x).detach().cpu().numpy()
        
    def forward(self, x):
        
#         x[:, 1, ...] = torch.log(x[:, 1, ...] + 1e-7)
        x1 = self.complex_conv1(x)
        if True in torch.isnan(x1).detach().cpu().numpy():
            st()
        x2 = self.proj2(x1)
        if True in torch.isnan(x2).detach().cpu().numpy():
            st()
        x3 = self.complex_conv2(x2)
        if True in torch.isnan(x3).detach().cpu().numpy():
            st()
        x4 = self.proj2(x3)
        if True in torch.isnan(x4).detach().cpu().numpy():
            st()
        x5 = self.linear_1(x4)
        if True in torch.isnan(x5).detach().cpu().numpy():
            st()
        
        x6 = self.conv_1(x5)
        if True in torch.isnan(x6).detach().cpu().numpy():
            for i in range(len(x6)):
                if True in torch.isnan(x6[i]).cpu().detach().numpy():
                    print(i)
            try_conv = self.conv_1(x5[164].unsqueeze(0))
            print(True in torch.isnan(try_conv).detach().cpu().numpy())
            try_conv = self.conv_1(x5[164:])
            print(True in torch.isnan(try_conv).detach().cpu().numpy())
            
            try_conv = self.conv_1(x5[163:])
            print(True in torch.isnan(try_conv).detach().cpu().numpy())
            st()
            
        x = self.bn_1(x6)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x_res = self.relu(x)
        
        x = self.id1(x_res) + self.res1(x_res)
        
        x = self.mp_1(x)
        x = self.conv_2(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x = self.bn_2(x)
        if True in torch.isnan(x).detach().cpu().numpy():
            st()
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        
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