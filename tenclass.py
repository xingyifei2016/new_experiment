# -*- coding: utf-8 -*-
import model as model_
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import random
import math
import os
import random
from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd
from pdb import set_trace as st
from logger import setup_logger

def sqrt_prep(data_dir, train_batch, test_batch, train_split, logger, validation_threshold=0.9):
    # sqrt(mag), cos(theta), sin(theta)
    # This is 10 - class
    data_x = []
    data_y = []
    for i, f in enumerate(listdir(data_dir)):
        if i % 1000 == 0:
            print("Loaded file "+str(i)+" of "+str(len(listdir(data_dir))))
        data = np.load(join(data_dir, f))
        label = f.split('_')[0].split('c')[1]
        if int(label)-1 <= 9:
            data_x.append(data)
            data_y.append(int(label)-1)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    xshape = data_x.shape
    data_x = data_x.reshape((xshape[0], xshape[1], xshape[2], xshape[3]))
    
    
#     # New data form
    mag = data_x[:, 4,...] 
    cos_ = data_x[:, 0,...] 
    sin_ = data_x[:, 1,...] 
    
    data_x[:, 0,...] = np.sqrt(mag)
    
    data_x[:, 1,...] = cos_
    data_x[:, 2,...] = sin_
    data_x = data_x[:, :3,...]
    
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))
    
    permutations = torch.randperm(int(len(data_y)))
    
    train_idx = permutations[:int(len(data_y) * train_split * validation_threshold)]
    validation_idx = permutations[int(len(data_y) * train_split * validation_threshold):int(len(data_y) * train_split)]
    test_idx = permutations[int(len(data_y) * train_split):]
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_val = torch.utils.data.Subset(data_set_11,indices=validation_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    params_train = {'batch_size': train_batch,
          'shuffle': True,
          'num_workers': 1}
    params_val = {'batch_size': test_batch,
              'shuffle': False,
              'num_workers': 1}
    
    logger.info(str(train_split * 100)+"% data used for training")
    logger.info(str(validation_threshold * 100)+"% of training data used for validation")
    train_generator = torch.utils.data.DataLoader(dataset=data_train, **params_train)
    val_generator = torch.utils.data.DataLoader(dataset=data_val, **params_train)
    test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
    return train_generator, val_generator, test_generator 
