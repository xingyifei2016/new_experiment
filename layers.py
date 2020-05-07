import torch
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import random
import math
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import make_interp_spline, BSpline
import scipy.interpolate as interpolate
import os
import random
from os import listdir
from os.path import isfile, join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import scipy.signal
from collections import Counter
from pdb import set_trace as st
from matplotlib.colors import hsv_to_rgb
import torch.nn.functional as f
import torch.nn.functional as F

eps = 0.000001

def weightNormalize1(weights):
    return ((weights**2)/torch.sum(weights**2))

def weightNormalize2(weights):
    return weights/torch.sum(weights**2)

def weightNormalize(weights, drop_prob=0.0):
    out = []
    for row in weights:
        if drop_prob==0.0:
            out.append(row**2/torch.sum(row**2))
        else:
            p = torch.randint(0, 2, (row.size())).float().cuda() 
            out.append((row**2/torch.sum(row**2))*p)
    return torch.stack(out)


class manifoldReLUv2angle(nn.Module):
    def __init__(self,channels):
        super(manifoldReLUv2angle, self).__init__()
        self.weight_abs = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.weight_cos = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.weight_sin = torch.nn.Parameter(torch.rand(1,channels), requires_grad=True)
        self.channels = channels 
        

    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        
        temp_abs = x[:,0,...]  
        temp_cos = x[:,1,...]
        temp_sin = x[:,2,...]
        
        temp_abs = (temp_abs.unsqueeze(1)+(weightNormalize1(self.weight_abs+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        
        
        temp_cos = (temp_cos.unsqueeze(1)*(weightNormalize2(self.weight_cos+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        
        temp_sin = (temp_sin.unsqueeze(1)*(weightNormalize2(self.weight_sin+eps)).unsqueeze(0).unsqueeze(3).unsqueeze(4).repeat(x_shape[0],1,1,x_shape[3],x_shape[4]))
        
        cos_sin = torch.cat((temp_cos, temp_sin),1)
        
        cos_sin = cos_sin / torch.sqrt(torch.sum(cos_sin ** 2, dim=1, keepdim=True))
        
        return torch.cat((temp_abs, cos_sin),1)
    
    

class ComplexConv(nn.Module):
    
    def __init__(self, in_channels, num_filters, kern_size, stride=(1, 1), num_tied_block=1, padding=0, dilation=1, groups=1):
        super(ComplexConv, self).__init__()
        
        # Convolution parameters
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if type(kern_size) == int:
            self.kern_size = (kern_size, kern_size)
        
        # Number of tied-block convolution kernels, 1 equals regular convolution.
        # Each tied-block convolution kernel goes through **in_channels // num_tied_block** channels
        self.num_blocks = num_tied_block
        
        if in_channels % num_tied_block != 0:
            assert("Number of tied block convolution needs to be multiple of in_channels: "+str(in_channels))
            
        if in_channels == 1 and num_tied_block != 1:
            assert("In channel of size 1 does not support tied-block convolution")
            
        self.in_channels = in_channels // num_tied_block
        
        # Out channels per blocks
        self.num_filters = num_filters
        self.out_channels = num_filters * num_tied_block
        # Initialize kernels
        self.mag_kernel = torch.nn.Parameter(torch.rand((self.num_filters, self.in_channels, self.kern_size[0], self.kern_size[1])), requires_grad=True)
        
        self.cos_kernel = torch.nn.Parameter(torch.rand((self.num_filters, self.in_channels, self.kern_size[0], self.kern_size[1])), requires_grad=True)
        
        self.sin_kernel = torch.nn.Parameter(torch.rand((self.num_filters, self.in_channels, self.kern_size[0], self.kern_size[1])), requires_grad=True)
        
        # Consistent with torch's initialization
        n = self.in_channels
        for k in self.kern_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        
        self.mag_kernel.data.uniform_(-stdv, stdv)
        self.cos_kernel.data.uniform_(-stdv, stdv)
        self.sin_kernel.data.uniform_(-stdv, stdv)
    def __repr__(self):
        return 'ComplexConv('+str(self.in_channels)+', '+str(self.out_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
        
    def forward(self, x):
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, out_channel, height, width]
        mag = x[:, 0, ...]
        cos_phase = x[:, 1, ...]
        sin_phase = x[:, 2, ...]
        
        # DO MAGNITUDE COMPONENT
        mag_kernel_sqrd = self.mag_kernel ** 2
        mag_kernel = mag_kernel_sqrd / torch.sum(torch.sum(mag_kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True)
        
        mag_list = [F.conv2d(mag[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
                             mag_kernel, None, self.stride, self.padding, self.dilation, \
                             self.groups) for i in range(self.num_blocks)]
        log_output = torch.cat(mag_list, dim=1)
        
        # DO COSINE COMPONENT
        cos_kernel_sqrd = self.cos_kernel ** 2
        cos_kernel = cos_kernel_sqrd / torch.sum(torch.sum(cos_kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True)
        
        cos_list = [F.conv2d(cos_phase[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
                             cos_kernel, None, self.stride, self.padding, self.dilation, \
                             self.groups) for i in range(self.num_blocks)]
        cos_output = torch.cat(cos_list, dim=1)
        
        # DO SINE COMPONENT
        sin_kernel_sqrd = self.sin_kernel ** 2
        sin_kernel = sin_kernel_sqrd / torch.sum(torch.sum(sin_kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True)
        
        sin_list = [F.conv2d(sin_phase[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
                             sin_kernel, None, self.stride, self.padding, self.dilation, \
                             self.groups) for i in range(self.num_blocks)]
        sin_output = torch.cat(sin_list, dim=1)
        
        # Need to normalize directional vectors such that
        # cos(theta)^2 + sin(theta)^2 = 1
        # First combine the cosine and sine outputs together
        # [Batch, 2, out_channel, height, width]
        cos_sin_output = torch.cat([cos_output.unsqueeze(1), sin_output.unsqueeze(1)], dim=1)
        
        # Compute sqrt(cos(theta)^2 + sin(theta)^2) for normalization
        cos_sin_magnitude = torch.sqrt(torch.sum(cos_sin_output ** 2, dim=1, keepdim=True))
        cos_sin_output = cos_sin_output / cos_sin_magnitude
        
        # Concate all final outputs together
        final_output = torch.cat([log_output.unsqueeze(1), cos_sin_output], dim=1)
        
        return final_output
    
class EuclideanConv(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1):
        super(EuclideanConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if type(kern_size) == int:
            self.kern_size = (kern_size, kern_size)
         
        # Only need one kernel. wEM(x, y)
        self.mag_kernel = torch.nn.Parameter(torch.rand((out_channels, in_channels, self.kern_size[0], self.kern_size[1])), requires_grad=True)
        
        # Consistent with torch's initialization
        n = self.in_channels
        for k in self.kern_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        
        self.mag_kernel.data.uniform_(-stdv, stdv)
        self.phase_kernel.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        # Input is of shape [B, 2, c, h, w]
        single_input = x # 0 is phase, 1 is magnitude
        
        phase = single_input[:, 0, ...]
        mag = single_input[:, 1, ...]

        
        mag_output = F.conv2d(mag, self.mag_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        phase_output = F.conv2d(phase, self.phase_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        final_output = torch.cat([phase_output.unsqueeze(1), mag_output.unsqueeze(1)], dim=1)
        return final_output
    
class DistanceTransform(nn.Module):
    def __init__(self, in_channels, kern_size, stride=(1,1), num_tied_block=1, padding=0, dilation=1, groups=1, b=1e-7):
        ## Magnitude do log / exp; Phase do weighted 
        super(DistanceTransform, self).__init__()
        
        self.in_channels = in_channels
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        if type(kern_size) == int:
            self.kern_size = (kern_size, kern_size)
            
        # Number of tied-block convolution kernels, 1 equals regular convolution.
        # Each tied-block convolution kernel goes through **in_channels // num_tied_block** channels
        self.num_blocks = num_tied_block
        
        if in_channels % num_tied_block != 0:
            assert("Number of tied block convolution needs to be multiple of in_channels: "+str(in_channels))
            
        
        self.b = b # For hyperparameter in distance calculation    
        
        self.in_channels = in_channels // num_tied_block
        
        # Initialize kernels
        self.mag_kernel = torch.nn.Parameter(torch.rand((self.in_channels, self.in_channels, \
                                                         self.kern_size[0], self.kern_size[1])), requires_grad=True)
        self.cos_kernel = torch.nn.Parameter(torch.rand((self.in_channels, self.in_channels, \
                                                           self.kern_size[0], self.kern_size[1])), requires_grad=True)
        self.sin_kernel = torch.nn.Parameter(torch.rand((self.in_channels, self.in_channels, \
                                                           self.kern_size[0], self.kern_size[1])), requires_grad=True)
        
        # Consistent with torch's initialization
        n = self.in_channels
        for k in self.kern_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        
        self.mag_kernel.data.uniform_(-stdv, stdv)
        self.cos_kernel.data.uniform_(-stdv, stdv)
        self.sin_kernel.data.uniform_(-stdv, stdv)
        
    def __repr__(self):
        return 'DistanceTransform('+str(self.in_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
    
    def forward(self, x):
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, in_channel, height, width]
        # The final output of this layer would be [Batch, in_channel, height, width]
        mag = x[:, 0, ...]
        cos_phase = x[:, 1, ...]
        sin_phase = x[:, 2, ...]
        
        # DO MAGNITUDE COMPONENT
        mag_kernel_sqrd = self.mag_kernel ** 2
        mag_kernel = mag_kernel_sqrd / torch.sum(torch.sum(mag_kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True)
        
        mag_list = [F.conv2d(mag[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
                             mag_kernel, None, self.stride, self.padding, self.dilation, \
                             self.groups) for i in range(self.num_blocks)]
        log_output = torch.cat(mag_list, dim=1)
        
        # DO COSINE COMPONENT
        cos_kernel_sqrd = self.cos_kernel ** 2
        cos_kernel = cos_kernel_sqrd / torch.sum(torch.sum(cos_kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True)
        
        cos_list = [F.conv2d(cos_phase[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
                             cos_kernel, None, self.stride, self.padding, self.dilation, \
                             self.groups) for i in range(self.num_blocks)]
        cos_output = torch.cat(cos_list, dim=1)
        
        # DO SINE COMPONENT
        sin_kernel_sqrd = self.sin_kernel ** 2
        sin_kernel = sin_kernel_sqrd / torch.sum(torch.sum(sin_kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True)
        
        sin_list = [F.conv2d(sin_phase[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
                             sin_kernel, None, self.stride, self.padding, self.dilation, \
                             self.groups) for i in range(self.num_blocks)]
        sin_output = torch.cat(sin_list, dim=1)
        
        # Need to normalize directional vectors such that
        # cos(theta)^2 + sin(theta)^2 = 1
        # First combine the cosine and sine outputs together
        # [Batch, 2, out_channel, height, width]
        cos_sin_output = torch.cat([cos_output.unsqueeze(1), sin_output.unsqueeze(1)], dim=1)
        
        # Compute sqrt(cos(theta)^2 + sin(theta)^2) for normalization
        cos_sin_magnitude = torch.sqrt(torch.sum(cos_sin_output ** 2, dim=1, keepdim=True))
        cos_sin_output = cos_sin_output / cos_sin_magnitude
        
        
        # For center-cropping original input
        output_xdim = cos_output.shape[2]
        output_ydim = cos_output.shape[3]
        input_xdim = x.shape[3]
        input_ydim = x.shape[4]
        
        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)
        
        cropped_input = x[:, :, :, start_x:start_x+output_xdim, start_y:start_y+output_ydim]
        
        # Compute distance according to sqrt(log^2[(|z2|+b)/(|z1+b|)] + acos^2(x^T * y))
        directional_distance = torch.sum(cos_sin_output * cropped_input[:, 1:, ...], dim=1) # [Batch, out_channe;, height, width]
        directional_distance = torch.acos(torch.clamp(directional_distance, -1+1e-5, 1-1e-5)) ** 2 ### Potential NaN problem
        
        magnitude_distance = (cropped_input[:, 0, ...] - log_output) ** 2
        
        return torch.sqrt(magnitude_distance + directional_distance)
        
        
        
        
#         upsampler = nn.Upsample(size=x_shape[-2:], mode='nearest')
        
#         # Upsample the respective outputs to the same size as inputs
#         mag_output = upsampler(mag_output)
#         cos_output = upsampler(cos_output)
#         sin_output = upsampler(sin_output)
        
#         # Take magnitude distance
#         if not self.take_log: 
#             mag_distance = torch.abs(torch.log(mag/(mag_output+eps)))
#         else:
#             mag_distance = torch.abs(torch.log(torch.exp(mag)/(mag_output+eps)))
            
#         phase_distance = torch.abs(phase-phase_output)
    
#         return mag_distance * (self.weight[0] **2) + phase_distance * (self.weight[1] **2)