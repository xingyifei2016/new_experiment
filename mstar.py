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
import resnet_loader as r
import tenclass as t
import resnet18 as re


def data_prep(data_dir, train_batch, test_batch, train_split, logger, validation_threshold=0.9):
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
    data_x = data_x.reshape((xshape[0], xshape[1], 1, xshape[2], xshape[3]))
    
    
#     # New data form
    mag = data_x[:, 4,...] + 0.5
    cos_ = data_x[:, 0,...] 
    sin_ = data_x[:, 1,...] 
    data_x[:, 0,...] = np.log(mag)
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






def test(model, device, test_loader, logger, epoch):
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            targets = target.cpu().numpy()
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    print("Test Accuracy is: "+str(100. * correct / len(test_loader.dataset)))
    logger.info("Test-"+str(epoch)+": "+str(100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def train(model, device, train_loader, optimizer, epoch, logger):
    train_acc = 0
    train_loss = 0
    for it,(local_batch, local_labels) in enumerate(train_loader):
        batch = torch.tensor(local_batch, requires_grad=True).cuda()
        labels = local_labels.cuda()
        optimizer.zero_grad()
        out = model(batch)
        _, predicted = torch.max(out, 1)
        total = labels.shape[0]
        train_acc += (predicted == labels).sum().item()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(out, labels)
        train_loss += loss 
        loss.backward()
        optimizer.step()
    print("#####EPOCH "+str(epoch)+"#####")
    print("Train accuracy is: "+str(train_acc / len(train_loader.dataset)*100.))
    print("Train loss is: "+str((train_loss / len(train_loader.dataset)*100.).item()))
    logger.info("Loss: "+str((train_loss / len(train_loader.dataset)*100.).item())+ " Train-"+str(epoch)+": "+str(train_acc / len(train_loader.dataset)*100.))
    
    
def compare_and_save(model, current_acc, highest, save_path, logger, batch, learn, split):
    res = current_acc > highest
    if res:
        if save_path is not None:
            try:
                os.remove(save_path+'.ckpt')
            except:
                pass
        highest = current_acc
        save_path = os.path.join('./save/', '[{acc}]-[{batch}]-[{learning_rate}]-11class-model-[{split}]'.format(acc = np.round(current_acc, 3), batch=batch, learning_rate=learn, split=split))
        torch.save(model.state_dict(), save_path+'.ckpt')
        logger.info('Saved model checkpoints into {}...'.format(save_path))
    return highest, save_path, res
        
def main():
    #argparse settings
    parser = argparse.ArgumentParser(description='PyTorch MSTAR Example') #400 and 0.001
    parser.add_argument('--batchsize', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 400)')
    parser.add_argument('--test_batchsize', type=int, default=400, metavar='N',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.015, metavar='LR',
                        help='learning rate (default: 0.015)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Adam momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default="./data_polar", metavar='N',
                        help='where data is stored')
    parser.add_argument('--use-pretrain', type=int, default=1, metavar='N',
                        help='Use pretrained model or not')
    
    try:
        os.mkdir('./log')
    except:
        pass
    
    # For model saving purposes, initializes as 0
    # If accuracy higher than "higher" then saves the model
    highest = 0
    
    # The actual path to save
    save_path=None
    
    try:
        os.mkdir('./save')
    except:
        pass
    
    
    
    logger = setup_logger('MSTAR logger')
    use_1517 = True
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     batches = [80, 50, 1000, 30, 600] 
    batches = [256]
    lrs = [0.015, 0.03, 0.005, 0.05]
    models = {'difference+tangent': model_.Dis_Tan, 'tangent only': model_.NL, 'difference': model_.Dis, 'tangent+difference': model_.Tan_Dis, 'previous_architecture': model_.ManifoldNetRes}
    num_repeat = 2
    splits = [0.1, 0.05, 0.01]
    
    
    
#     #### FOR BASELINE RESNET ONLY ####
#     data_preps = {'r_cos(theta)_sin(theta)': r.data_prep_baseline1, '(log(r+1), cos(theta),sin(theta))': r.data_prep_baseline4, '(log(r+0.1), cos(theta),sin(theta))': r.data_prep_baseline5}
    
#     corresponding_models = {'r_cos(theta)_sin(theta)': model_.ResNet3, '(log(r+1), cos(theta),sin(theta))': model_.ResNet3, '(log(r+0.1), cos(theta),sin(theta))': model_.ResNet3}
    
#     data_preps = {'(log(r+0.01), cos(theta),sin(theta))': r.data_prep_baseline6, 'real_imag':r.data_prep_baseline2, '(sqrt(r), cos(theta),sin(theta))': r.data_prep_baseline3, "r_theta": r.data_prep_baseline}
    
#     corresponding_models = {'(log(r+0.01), cos(theta),sin(theta))': model_.ResNet3, 'real_imag':model_.ResNet, '(sqrt(r), cos(theta),sin(theta))': model_.ResNet3, "r_theta": model_.ResNet}
    
#     for prep in data_preps.keys():
        
#         model = corresponding_models[prep]().cuda()
#         logger.info(model)
#         model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         logger.info("#Model Parameters: "+str(params))
#         for split in splits:
#             for i in batches:
#                 train_loader, val_loader, test_loader = data_preps[prep](args.data_dir, i, args.test_batchsize, split, logger)
                
#                 for j in lrs:
#                     even_higher = 0
#                     for ee in range(num_repeat):
#                         optimizer = optim.Adam(model.parameters(), lr=j, eps=1e-8, amsgrad=True)
#                         logger.info("Learning Rate: "+str(j))
#                         logger.info("Batch Size: "+str(i))
#                         logger.info(prep)
#                         logger.info("Repeating trial "+str(ee))
#                         logger.info("Split "+str(split))
#                         highest = 0

#                         for epoch in range(1, args.epochs + 1):
#                             train(model, device, train_loader, optimizer, epoch, logger)
#                             acc=test(model, device, val_loader, logger, epoch)
#                             highest, save_path, new_best = compare_and_save(model, acc, highest, save_path, logger, i, j, split)

#                             if new_best:
#                                 logger.info("####NEW TEST RESULT#####")
#                                 acc=test(model, device, test_loader, logger, epoch)
#                                 logger.info("####NEW TEST RESULT#####")
#                                 even_higher = acc
#                         acc=test(model, device, test_loader, logger, epoch)
#                         even_higher = max(acc, even_higher)
#                         logger.info("########## NEW MODEL ###########")
#                         model = corresponding_models[prep]().cuda()
#                     # This is accuracy for best validation model
#                     logger.info("####Highest Testing Accuracy: "+str(even_higher))
    

    
    
    
   
    
    
    
    # For running experiments
    for model1 in models.keys():
        model = models[model1]().cuda()
        logger.info(model)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("#Model Parameters: "+str(params))
        for split in splits:
            for i in batches:
                train_loader, val_loader, test_loader = data_prep(args.data_dir, i, args.test_batchsize, split, logger)
                
                for j in lrs:
                    even_higher = 0
                    for ee in range(num_repeat):
                        optimizer = optim.Adam(model.parameters(), lr=j, eps=1e-8, amsgrad=True)
                        logger.info("Learning Rate: "+str(j))
                        logger.info("Batch Size: "+str(i))
                        logger.info(model1)
                        logger.info("Repeating trial "+str(ee))
                        logger.info("Split "+str(split))
                        highest = 0

                        for epoch in range(1, args.epochs + 1):
                            train(model, device, train_loader, optimizer, epoch, logger)
                            acc=test(model, device, val_loader, logger, epoch)
                            highest, save_path, new_best = compare_and_save(model, acc, highest, save_path, logger, i, j, split)

                            if new_best:
                                logger.info("####NEW TEST RESULT#####")
                                acc=test(model, device, test_loader, logger, epoch)
                                logger.info("####NEW TEST RESULT#####")
                                even_higher = max(acc, even_higher)
                        acc=test(model, device, test_loader, logger, epoch)
                        even_higher = max(acc, even_higher)
                        logger.info("########## NEW MODEL ###########")
                        model = models[model1]().cuda()
                    logger.info("####Highest Testing Accuracy: "+str(even_higher))
                    
                    
    model = re.ResNet().cuda()
    logger.info(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("#Model Parameters: "+str(params))
    for split in splits:
        for i in batches:
            train_loader, val_loader, test_loader = t.sqrt_prep(args.data_dir, i, args.test_batchsize, split, logger)

            for j in lrs:
                even_higher = 0
                for ee in range(num_repeat):
                    optimizer = optim.Adam(model.parameters(), lr=j, eps=1e-8, amsgrad=True)
                    logger.info("Learning Rate: "+str(j))
                    logger.info("Batch Size: "+str(i))
                    logger.info("ResNet18")
                    logger.info("Repeating trial "+str(ee))
                    logger.info("Split "+str(split))
                    highest = 0

                    for epoch in range(1, args.epochs + 1):
                        train(model, device, train_loader, optimizer, epoch, logger)
                        acc=test(model, device, val_loader, logger, epoch)
                        highest, save_path, new_best = compare_and_save(model, acc, highest, save_path, logger, i, j, split)

                        if new_best:
                            logger.info("####NEW TEST RESULT#####")
                            acc=test(model, device, test_loader, logger, epoch)
                            logger.info("####NEW TEST RESULT#####")
                            even_higher = max(acc, even_higher)
                    acc=test(model, device, test_loader, logger, epoch)
                    even_higher = max(acc, even_higher)
                    logger.info("########## NEW MODEL ###########")
                    model = re.ResNet().cuda()
                logger.info("####Highest Testing Accuracy: "+str(even_higher))

if __name__ == '__main__':
    main()
       

