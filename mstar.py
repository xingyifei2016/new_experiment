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

from models import new_models as m
from dataloaders import tenclass as t
import matplotlib.pyplot as plt
from models import resnet18 as re
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('books_read.png')
#     st()
    
    

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
        plot_grad_flow(model.named_parameters())
        
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
        save_path = os.path.join('./save/', '[{acc}]-[{batch}]-[{learning_rate}]-[{split}]'.format(acc = np.round(current_acc, 3), batch=batch, learning_rate=learn, split=split))
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
    lrs = [0.001, 0.0005, 0.01, 0.005, 0.05]
#     models = {'difference+tangent': model_.Dis_Tan, 'tangent only': model_.NL, 'difference': model_.Dis, 'tangent+difference': model_.Tan_Dis, 'previous_architecture': model_.ManifoldNetRes}
    models = {'newModels': m.Model1}
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
                train_loader, val_loader, test_loader = t.data_prep(args.data_dir, i, args.test_batchsize, split, logger)
                
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
       

