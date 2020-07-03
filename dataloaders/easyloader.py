# -*- coding: utf-8 -*-
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
from pdb import set_trace as st
from logger import setup_logger

def data_prep(data_dir, train_batch, test_batch, train_split, logger, validation_threshold=0.9):
    # log(mag), cos(theta), sin(theta)
    # This is 2 - class
    data_x = []
    data_y = []
    ellipses = np.load(data_dir+'/ellipses_inside.npy')
    rectangles = np.load(data_dir+'/rectangles_inside.npy')
    
    ellipse_labels = np.zeros(len(ellipses))
    rectangle_labels = np.ones(len(rectangles))
    
    ellipses_mag = np.expand_dims(np.log(ellipses[:, 0, ...]), axis=1)
    ellipses_cos = np.expand_dims(np.cos(ellipses[:, 1, ...]), axis=1)
    ellipses_sin = np.expand_dims(np.sin(ellipses[:, 1, ...]), axis=1)
    
    ellipses_final = np.concatenate((ellipses_mag, ellipses_cos, ellipses_sin), axis=1)
    
    rectangles_mag = np.expand_dims(np.log(rectangles[:, 0, ...]), axis=1)
    rectangles_cos = np.expand_dims(np.cos(rectangles[:, 1, ...]), axis=1)
    rectangles_sin = np.expand_dims(np.sin(rectangles[:, 1, ...]), axis=1)
    
    rectangles_final = np.concatenate((rectangles_mag, rectangles_cos, rectangles_sin), axis=1)
    
    data_x = np.concatenate((ellipses_final, rectangles_final), axis=0)
    data_y = np.concatenate((ellipse_labels, rectangle_labels), axis=0)
    
    perm_list = np.random.permutation(int(len(data_x)))

    data_x = data_x[perm_list]
    data_y = data_y[perm_list]
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


def data_prep_random(data_dir, train_batch, test_batch, train_split, logger, validation_threshold=0.9):
    # This tests against uniform random vs. ellipses/rectangles
    # log(mag), cos(theta), sin(theta)
    # This is 2 - class
    data_x = []
    data_y = []
#     ellipses = np.load(data_dir+'/ellipses.npy')
    rectangles = np.load(data_dir+'/rectangles.npy')#np.random.uniform(size=ellipses.shape)
    ellipses = np.random.uniform(size=rectangles.shape)
    
    ellipse_labels = np.zeros(len(ellipses))
    rectangle_labels = np.ones(len(rectangles))
    
    ellipses_mag = np.expand_dims(np.log(ellipses[:, 0, ...]), axis=1)
    ellipses_cos = np.expand_dims(np.cos(ellipses[:, 1, ...]), axis=1)
    ellipses_sin = np.expand_dims(np.sin(ellipses[:, 1, ...]), axis=1)
    
    ellipses_final = np.concatenate((ellipses_mag, ellipses_cos, ellipses_sin), axis=1)
    
    rectangles_mag = np.expand_dims(np.log(rectangles[:, 0, ...]), axis=1)
    rectangles_cos = np.expand_dims(np.cos(rectangles[:, 1, ...]), axis=1)
    rectangles_sin = np.expand_dims(np.sin(rectangles[:, 1, ...]), axis=1)
    
    rectangles_final = np.concatenate((rectangles_mag, rectangles_cos, rectangles_sin), axis=1)
    
    data_x = np.concatenate((ellipses_final, rectangles_final), axis=0)
    data_y = np.concatenate((ellipse_labels, rectangle_labels), axis=0)
    
    perm_list = np.random.permutation(int(len(data_x)))

    data_x = data_x[perm_list]
    data_y = data_y[perm_list]
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

def data_prep_one(data_dir, train_batch, test_batch, train_split, logger, validation_threshold=0.9):
    # Training set is just one image from each class
    # log(mag), cos(theta), sin(theta)
    # This is 2 - class
    data_x = []
    data_y = []
    ellipses = np.load(data_dir+'/ellipses.npy')
    rectangles = np.load(data_dir+'/rectangles.npy')
    
    ellipse_labels = np.zeros(len(ellipses))
    rectangle_labels = np.ones(len(rectangles))
    
    ellipses_mag = np.expand_dims(np.log(ellipses[:, 0, ...]), axis=1)
    ellipses_cos = np.expand_dims(np.cos(ellipses[:, 1, ...]), axis=1)
    ellipses_sin = np.expand_dims(np.sin(ellipses[:, 1, ...]), axis=1)
    
    ellipses_final = np.concatenate((ellipses_mag, ellipses_cos, ellipses_sin), axis=1)
    
    rectangles_mag = np.expand_dims(np.log(rectangles[:, 0, ...]), axis=1)
    rectangles_cos = np.expand_dims(np.cos(rectangles[:, 1, ...]), axis=1)
    rectangles_sin = np.expand_dims(np.sin(rectangles[:, 1, ...]), axis=1)
    
    rectangles_final = np.concatenate((rectangles_mag, rectangles_cos, rectangles_sin), axis=1)
    
    data_x = np.concatenate((ellipses_final, rectangles_final), axis=0)
    data_y = np.concatenate((ellipse_labels, rectangle_labels), axis=0)
    
    perm_list = np.random.permutation(int(len(data_x)))

    data_x = data_x[perm_list]
    data_y = data_y[perm_list]
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))
    
    permutations = torch.randperm(int(len(data_y)))
    
    train_idx = permutations[:int(len(data_y) * train_split * validation_threshold)]
    validation_idx = permutations[int(len(data_y) * train_split * validation_threshold):int(len(data_y) * train_split)]
    test_idx = permutations[int(len(data_y) * train_split):]
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_val = torch.utils.data.Subset(data_set_11,indices=validation_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    
    data_train = torch.utils.data.TensorDataset(torch.from_numpy(np.concatenate((ellipses_final[np.newaxis, 0], rectangles_final[np.newaxis, 0]), axis=0)).type(torch.FloatTensor), torch.from_numpy (np.concatenate((ellipse_labels[np.newaxis, 0], rectangle_labels[np.newaxis, 0]), axis=0)).type(torch.LongTensor))
    
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