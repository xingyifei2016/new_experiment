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

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import scipy.signal
from collections import Counter
from pdb import set_trace as st
from matplotlib.colors import hsv_to_rgb
import torch.nn.functional as f
import torch.nn.functional as F

eps = 1e-6

def m(x): #TODO Utkarsh rename this
    # Debugging function
    is_nan = True in torch.isnan(x).cpu().detach().numpy() 
    is_inf = (torch.min(x) == torch.tensor(float("-Inf"))) or (torch.max(x) == torch.tensor(float("Inf")))
    return is_nan or is_inf

def weightNormalize1(weights): #TODO Utkarsh these functions need to be renamed
    # Function used by Rudra's G-transport
    return ((weights**2)/torch.sum(weights**2)) #TODO Utkarsh: Which dimensions? Seems to be adding all dimensions up

def weightNormalize2(weights):
    # Function used by Rudra's G-transport
    return weights/torch.sum(weights**2)


class manifoldReLUv2angle(nn.Module): #TODO Utkarsh: look at this
    # Rudra's implementation of G-Transport
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
    
    
class Cylindrical_Nonlinearity(torch.autograd.Function):
    # This is an autograd function for the tangent-ReLU function
    # Checks for the cylindrical part only (the log-magnitude portion is done by calling ReLU)
    # Forward pass: given [cos(theta), sin(theta)], sets value to [1, 0] if sin(theta) < 0, else remain unchanged
    # Backward pass: if previous input has sin(theta) < 0, set gradient to 0, else gradient remains unchanged
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    
    @staticmethod
    def forward(ctx, inputs):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        
        ctx.save_for_backward(inputs)
        
        temp_cos = inputs[:,0,...]
        temp_sin = inputs[:,1,...]
        
        temp_cos[temp_sin <= 0] = 1
        temp_sin[temp_sin <= 0] = 0
        cos_sin = torch.cat((temp_cos.unsqueeze(1), temp_sin.unsqueeze(1)), 1)
        
        
        return cos_sin
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        
        inputs, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        grad_input_cos = grad_input[:,0,...]
        grad_input_sin = grad_input[:,1,...]
        
        
        grad_input_cos[inputs[:, 1, ...] <= 0] = 0
        grad_input_sin[inputs[:, 1, ...] <= 0] = 0
        
        
        return torch.cat((grad_input_cos.unsqueeze(1), grad_input_sin.unsqueeze(1)), 1)
    
    
    
class tangentRELU(nn.Module):
    def __init__(self):
        # Applies tangent reLU to inputs.
        super(tangentRELU, self).__init__()
        self.relu = nn.ReLU()
        
        #This part applies the previously defined cylindrical nonlinearity
        self.cn = Cylindrical_Nonlinearity.apply
        
    def forward(self, x):
        #Shape: [batches, features, in_channels, spatial_x, spatial_y]
        x_shape = x.shape  
        
        temp_abs = x[:,0,...]  
        temp_cos = x[:,1,...]
        temp_sin = x[:,2,...]
        
        # For log-space magnitude is just reLU
        final_abs = self.relu(temp_abs).unsqueeze(1)
        
        # For cylindrical coordinates, if y direction < 0, converts to (1, 0); else remain the same
        cos_sin = torch.cat((temp_cos.unsqueeze(1), temp_sin.unsqueeze(1)), 1)
        
        cos_sin = self.cn(cos_sin)
        return torch.cat((final_abs, cos_sin),1)   
    
    
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
        self.weight = torch.nn.Parameter(torch.rand((self.num_filters, self.in_channels, self.kern_size[0], self.kern_size[1])), requires_grad=True)
        
        # Consistent with torch's initialization
        n = self.in_channels #TODO Utkarsh: Seems like we are initializing weights manually. Use nn.init functions?
        for k in self.kern_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        
        self.weight.data.uniform_(-stdv, stdv)
        
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
        
        # Normalize kernel
        kernel_sqrd = self.weight ** 2
        kernel = kernel_sqrd / torch.sum(torch.sum(torch.sum(kernel_sqrd, dim=2, keepdim=True), dim=3, keepdim=True), dim=1, keepdim=True) #TODO Utkarsh: do dim=(1,2,3)?
        
        # DO MAGNITUDE COMPONENT
        mag_shape = mag.shape
        mag = mag.reshape((-1, self.in_channels, mag_shape[-2], mag_shape[-1])) #TODO Utkarsh: Why this reshape? Shouldn't input shape already be [batch x channels x H x W]?
        log_output = F.conv2d(mag, weight=kernel, \
                             stride=self.stride, padding=self.padding, \
                             dilation=self.dilation, groups = self.groups)
        log_output = log_output.reshape(mag.shape[0], self.num_filters*self.num_blocks, log_output.shape[-2], log_output.shape[-1])
        
        
        # DO COSINE COMPONENT
        cos_phase_shape = cos_phase.shape
        cos_phase = cos_phase.reshape((-1, self.in_channels, cos_phase_shape[-2], cos_phase_shape[-1]))
        cos_output = F.conv2d(cos_phase, weight=kernel, \
                             stride=self.stride, padding=self.padding, \
                             dilation=self.dilation, groups=self.groups)
        cos_output = cos_output.reshape(cos_phase.shape[0], self.num_filters*self.num_blocks, cos_output.shape[-2], cos_output.shape[-1])
#         cos_list = [F.conv2d(cos_phase[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
#                              cos_kernel, None, self.stride, self.padding, self.dilation, \
#                              self.groups) for i in range(self.num_blocks)]
#         cos_output = torch.cat(cos_list, dim=1)
        
        # DO SINE COMPONENT
        
        # [Batch, total_channels, H, W] ----> [Batch*L, kernel_channels, H, W]
        sin_phase_shape = sin_phase.shape
        sin_phase = sin_phase.reshape((-1, self.in_channels, sin_phase_shape[-2], sin_phase_shape[-1]))
        sin_output = F.conv2d(sin_phase, weight=kernel, \
                             stride=self.stride, padding=self.padding, \
                             dilation=self.dilation, groups=self.groups)
        sin_output = sin_output.reshape(sin_phase.shape[0], self.num_filters*self.num_blocks, sin_output.shape[-2], sin_output.shape[-1])
        
#         sin_list = [F.conv2d(sin_phase[:, i*self.in_channels:(i+1)*self.in_channels, ...], \
#                              sin_kernel, None, self.stride, self.padding, self.dilation, \
#                              self.groups) for i in range(self.num_blocks)]
#         sin_output = torch.cat(sin_list, dim=1)
        
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
        if m(final_output): #TODO Utkarsh: Better function names for error detection
            st()
        return final_output
    

    # def forward(self, x):
    #     # How I (Utkarsh) might write this function
    #     N, CC, C, H, W = x.shape # Batch, 3, channels, H, W
    #     x = x.reshape(-1, self.in_channels, H, W)

    #     kernel_sqrd = torch.square(self.kernel)
    #     kernel = kernel_sqrd / torch.sum(kernel_sqrd, dim=(1,2,3), keepdim=True)
        
    #     conv_out = F.conv2d(x, weight=kernel, stride=self.stride, 
    #                         padding=self.padding, dilation=self.dilation, groups=self.groups)
    #     conv_out = conv_out.reshape(N, 3, self.num_filters*self.num_blocks , H, W)
        
    #     phase_norm = torch.norm(conv_out[:,1:],dim=1,keepdim=True)
        
    #     return torch.cat([[conv_out[:,:1], conv_out[:,1:]/phase_norm]], dim=1)

    
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
        
        self.wFM = ComplexConv(in_channels=self.in_channels, num_filters=1, kern_size=self.kern_size, stride=self.stride, num_tied_block=self.num_blocks)
        
    def __repr__(self):
        return 'DistanceTransform('+str(self.in_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
    
    def forward(self, x):
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        wFMs = self.wFM(x)
        wFMs = wFMs.repeat(1, 1, self.in_channels, 1, 1)
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, in_channel, height, width]
        # The final output of this layer would be [Batch, in_channel, height, width]
        log_output = wFMs[:, 0, ...]
        cos_output = wFMs[:, 1, ...]
        sin_output = wFMs[:, 2, ...]
        cos_sin_output = wFMs[:, 1:, ...]
        
        
        # For center-cropping original input
        output_xdim = cos_output.shape[2]
        output_ydim = cos_output.shape[3]
        input_xdim = x.shape[3]
        input_ydim = x.shape[4]
        
        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)
        
        cropped_input = x[:, :, :, start_x:start_x+output_xdim, start_y:start_y+output_ydim] #TODO Utkarsh: Think this through
        
        #TODO Utkarsh add checks for norms of the different phases. acos only gives nans if the norms are larger than 1.

        # Compute distance according to sqrt(log^2[(|z2|+b)/(|z1+b|)] + acos^2(x^T * y))
        directional_distance = torch.sum(cos_sin_output * cropped_input[:, 1:, ...], dim=1) # [Batch, out_channe;, height, width]
        directional_distance = torch.acos(torch.clamp(directional_distance, -1+1e-5, 1-1e-5)) ### Potential NaN problem

        magnitude_distance = cropped_input[:, 0, ...] - log_output 
        
        return torch.sqrt(magnitude_distance**2 + directional_distance**2)
    

class SignedDistanceTransform(nn.Module):
    def __init__(self, in_channels, kern_size, stride=(1,1), num_tied_block=1, padding=0, dilation=1, groups=1, b=1e-7):
        ## Magnitude do log / exp; Phase do weighted 
        super(SignedDistanceTransform, self).__init__()
        
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
        
        self.wFM = ComplexConv(in_channels=self.in_channels, num_filters=self.in_channels, kern_size=self.kern_size, stride=self.stride, num_tied_block=self.num_blocks)
        
    def __repr__(self):
        return 'DistanceTransform('+str(self.in_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
    
    def forward(self, x):
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        wFMs = self.wFM(x)
        
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, in_channel, height, width]
        # The final output of this layer would be [Batch, in_channel, height, width]
        log_output = wFMs[:, 0, ...]
        cos_output = wFMs[:, 1, ...]
        sin_output = wFMs[:, 2, ...]
        cos_sin_output = wFMs[:, 1:, ...]
        
        
        # For center-cropping original input
        output_xdim = cos_output.shape[2]
        output_ydim = cos_output.shape[3]
        input_xdim = x.shape[3]
        input_ydim = x.shape[4]
        
        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)
        
        cropped_input = x[:, :, :, start_x:start_x+output_xdim, start_y:start_y+output_ydim] #TODO Utkarsh: Think this through
        
        #TODO Utkarsh add checks for norms of the different phases. acos only gives nans if the norms are larger than 1.

        # Compute distance according to sqrt(log^2[(|z2|+b)/(|z1+b|)] + acos^2(x^T * y))
        directional_distance = torch.sum(cos_sin_output * cropped_input[:, 1:, ...], dim=1) # [Batch, out_channe;, height, width]
        directional_distance = torch.acos(torch.clamp(directional_distance, -1+1e-5, 1-1e-5)) ### Potential NaN problem

        assert cos_sin_output.shape[1] == 2
        assert (cos_sin_output.norm(dim=1) - 1).abs().max() < 1e-5
        assert cropped_input[:,1:].shape[1] == 2
        assert (cropped_input[:,1:].norm(dim=1) - 1).abs().max() < 1e-5

        cos_sin_output_aug = torch.cat([cos_sin_output, cos_sin_output[:,1:]*0.0], dim=1)
        cropped_input_aug = torch.cat([cropped_input[:,1:], cropped_input[:,:1]*0.0], dim=1)

        cross = torch.sign(torch.cross(cos_sin_output_aug, cropped_input_aug, dim=1)[-1])
        
        assert cross.shape[1] == 1


        magnitude_distance = cropped_input[:, 0, ...] - log_output 
        
        l2_dist = torch.sqrt(magnitude_distance**2 + directional_distance**2)

        assert l2_dist.shape == cross.shape
        return l2_dist*cross


class DistanceTransformUpsample(nn.Module): #TODO Utkarsh look at this
    def __init__(self, in_channels, kern_size, stride=(1,1), num_tied_block=1, padding=0, dilation=1, groups=1, b=1e-7):
        ## Magnitude do log / exp; Phase do weighted 
        super(DistanceTransformUpsample, self).__init__()
        
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
        
        self.wFM = ComplexConv(in_channels=self.in_channels, num_filters=self.in_channels, kern_size=self.kern_size, stride=self.stride, num_tied_block=self.num_blocks)
        
    def __repr__(self):
        return 'DistanceTransform('+str(self.in_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
    
    def forward(self, x):
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        wFMs = self.wFM(x)
        
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, in_channel, height, width]
        # The final output of this layer would be [Batch, in_channel, height, width]
        log_output = wFMs[:, 0, ...]
        cos_output = wFMs[:, 1, ...]
        sin_output = wFMs[:, 2, ...]
        cos_sin_output = wFMs[:, 1:, ...]
        
        # For center-cropping original input
        output_xdim = cos_output.shape[2]
        output_ydim = cos_output.shape[3]
        input_xdim = x.shape[3]
        input_ydim = x.shape[4]
        
        m = nn.Upsample(size=(input_xdim, input_ydim), mode='bilinear')
        
        
        
        cropped_input = x
        log_output = m(log_output)
        cos_sin_output = torch.cat((m(cos_sin_output[:, 0, ...]).unsqueeze(1), m(cos_sin_output[:, 1, ...]).unsqueeze(1)), dim=1) #TODO Phase upsample should be followed by normalization
        
        # Compute distance according to sqrt(log^2[(|z2+b|)/(|z1+b|)] + acos^2(x^T * y))
        directional_distance = torch.sum(cos_sin_output * cropped_input[:, 1:, ...], dim=1) # [Batch, out_channe;, height, width]
        directional_distance = torch.acos(torch.clamp(directional_distance, -1+1e-5, 1-1e-5)) ### Potential NaN problem
        
        magnitude_distance = cropped_input[:, 0, ...] - log_output
        
        return torch.sqrt(magnitude_distance**2 + directional_distance**2)
        
        
class DifferenceLayer(nn.Module):
    def __init__(self, in_channels, kern_size, stride=(1,1), num_tied_block=1, padding=0, dilation=1, groups=1, b=1e-7):
        ## Magnitude do log / exp; Phase do weighted 
        super(DifferenceLayer, self).__init__()
        
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
        
         
        self.wFM = ComplexConv(in_channels=self.in_channels, num_filters=1, kern_size=self.kern_size, stride=self.stride, num_tied_block=self.num_blocks)
        
    def __repr__(self):
        return 'DistanceLayer('+str(self.in_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
    
    def forward(self, x):
        
        wFMs = self.wFM(x)
        wFMs = wFMs.repeat(1, 1, self.in_channels, 1, 1)
        
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, in_channel, height, width]
        # The final output of this layer would be [Batch, in_channel, height, width]
        log_output = wFMs[:, 0, ...]
        cos_output = wFMs[:, 1, ...]
        sin_output = wFMs[:, 2, ...]
        cos_sin_output = wFMs[:, 1:, ...]
        
        # For center-cropping original input
        output_xdim = cos_output.shape[2]
        output_ydim = cos_output.shape[3]
        input_xdim = x.shape[3]
        input_ydim = x.shape[4]
        
        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)
        
        cropped_input = x[:, :, :, start_x:start_x+output_xdim, start_y:start_y+output_ydim]
        
        # Compute distance according to sqrt(log^2[(|z2|+b)/(|z1|+b)] + acos^2(x^T * y))
        # Need to add noise or else normalization may have zero entries which cause NaN
        directional_difference = cropped_input[:, 1:, ...] - cos_sin_output
        # directional_difference = (directional_difference + direction_noise) / torch.sqrt(torch.sum(directional_difference ** 2 + direction_noise, dim=1, keepdim=True)) #TODO Utkarsh fix this
        
        directional_difference = directional_difference/(torch.norm(directional_difference, dim=1)+eps)
        
        if m(directional_difference):
            st()
        magnitude_difference = cropped_input[:, 0, ...] - log_output
        
        return torch.cat((magnitude_difference.unsqueeze(1), directional_difference), dim=1)        
        
class DifferenceLayerUpsample(nn.Module): #TODO Utkarsh look at this
    def __init__(self, in_channels, kern_size, stride=(1,1), num_tied_block=1, padding=0, dilation=1, groups=1, b=1e-7):
        ## Magnitude do log / exp; Phase do weighted 
        super(DifferenceLayerUpsample, self).__init__()
        
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
        
        self.wFM = ComplexConv(in_channels=self.in_channels, num_filters=self.in_channels, kern_size=self.kern_size, stride=self.stride, num_tied_block=self.num_blocks)
        
    def __repr__(self):
        return 'DistanceLayer('+str(self.in_channels)+', kernel_size='+str(self.kern_size)+', stride='+str(self.stride)+', num_tied_blocks='+str(self.num_blocks)+')'    
    
    def forward(self, x):
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        wFMs = self.wFM(x)
        
        # Input is of shape [Batch, 3, channel, height, width]
        # Cylindrical representation of data: [log(|z|), x/|z|, y/|z|]
        
        # Separate each component and do convolution along each component
        # The input of each component is [Batch, in_channel, height, width]
        # The output of each component is [Batch, in_channel, height, width]
        # The final output of this layer would be [Batch, in_channel, height, width]
        mag_output = wFMs[:, 0, ...]
        cos_output = wFMs[:, 1, ...]
        sin_output = wFMs[:, 2, ...]
        cos_sin_output = wFMs[:, 1:, ...]
        
        
        # For center-cropping original input
        output_xdim = cos_output.shape[2]
        output_ydim = cos_output.shape[3]
        input_xdim = x.shape[3]
        input_ydim = x.shape[4]
        
        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)
        
        m = nn.Upsample(size=(input_xdim, input_ydim), mode='bilinear')
        
        
        
        cropped_input = x
        log_output = m(log_output)
        cos_sin_output = torch.cat((m(cos_sin_output[:, 0, ...]).unsqueeze(1), m(cos_sin_output[:, 1, ...]).unsqueeze(1)), dim=1)
        # Compute distance according to sqrt(log^2[(|z2|+b)/(|z1|+b)] + acos^2(x^T * y))
        # Need to add noise or else normalization may have zero entries which cause NaN
        # Compute distance according to sqrt(log^2[(|z2|+b)/(|z1|+b)] + acos^2(x^T * y))
        # Need to add noise or else normalization may have zero entries which cause NaN
        direction_noise = 1e-5
        directional_difference = cropped_input[:, 1:, ...] - cos_sin_output
        directional_difference = (directional_difference + direction_noise) / torch.sqrt(torch.sum(directional_difference ** 2 + direction_noise, dim=1, keepdim=True))
        
        if m(directional_difference):
            st()
        magnitude_difference = cropped_input[:, 0, ...] - log_output
        
        return torch.cat((magnitude_difference.unsqueeze(1), directional_difference), dim=1)  
    
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
        self.phase_kernel.data.uniform_(-stdv, stdv) #TODO Utkarsh remove phase kernel
    
    def forward(self, x):
        # Input is of shape [B, 2, c, h, w]
        single_input = x # 0 is phase, 1 is magnitude
        
        phase = single_input[:, 0, ...]
        mag = single_input[:, 1, ...]

        
        mag_output = F.conv2d(mag, self.mag_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        phase_output = F.conv2d(phase, self.phase_kernel, None, self.stride, self.padding, self.dilation, self.groups)

        final_output = torch.cat([phase_output.unsqueeze(1), mag_output.unsqueeze(1)], dim=1)
        return final_output