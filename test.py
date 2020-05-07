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

def data_prep(data_dir, train_batch, test_batch, use_1517, logger):
    data_x = []
    data_y = []
    for i, f in enumerate(listdir(data_dir)):
        if i % 1000 == 0:
            print("Loaded file "+str(i)+" of "+str(len(listdir(data_dir))))
        data = np.load(join(data_dir, f))
        label = f.split('_')[0].split('c')[1]
        data_x.append(data)
        data_y.append(int(label)-1)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    xshape = data_x.shape
    data_x = data_x.reshape((xshape[0], xshape[1], 1, xshape[2], xshape[3]))
    
    mag = data_x[:, 4,...] + 1e-7
    cos_ = data_x[:, 0,...] 
    sin_ = data_x[:, 1,...] 
    data_x[:, 0,...] = np.log(mag)
    data_x[:, 1,...] = cos_
    data_x[:, 2,...] = sin_
    data_x = data_x[:, :3,...]
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))
    train_idx, test_idx = index_split(use_1517, logger)
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    params_train = {'batch_size': train_batch,
          'shuffle': True,
          'num_workers': 1}
    params_val = {'batch_size': test_batch,
              'shuffle': False,
              'num_workers': 1}
    train_generator = torch.utils.data.DataLoader(dataset=data_train, **params_train)
    test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
    return train_generator, test_generator 

def index_split(use_1517, logger):
    #Splitting method for our MSTAR data
    #If use_1517 is True, use the 15/17 depression split
    #If use_1517 is False, use the Seen/Unseen data split
    
    csv_path = './chipinfo.csv' 
    df = pd.read_csv(csv_path)
    training = df.loc[df['depression'] == 17]
    subclass_9 = training.loc[training['target_type'] != 'bmp2_tank']
    subclass_8 = subclass_9.loc[subclass_9['target_type'] != 't72_tank'].index.values
    class_1_train = np.array(training.loc[training['serial_num']=='c21'].index.values)
    class_3_train = np.array(training.loc[training['serial_num']=='132'].index.values)
    subclass = np.concatenate([subclass_8, class_1_train, class_3_train], axis=0)
    training = training.index
    testing = df.loc[df['depression'] == 15]
    subclass_test9 = testing.loc[testing['target_type'] != 'bmp2_tank']
    subclass_test8 = np.array(subclass_test9.loc[subclass_test9['target_type']=='t72_tank'].index.values)
    class_1_test2 = np.array(testing.loc[testing['serial_num']=='9563'].index.values)
    class_1_test3 = np.array(testing.loc[testing['serial_num']=='9566'].index.values)
    class_3_test2 = np.array(testing.loc[testing['serial_num']=='812'].index.values)
    class_3_test3 = np.array(testing.loc[testing['serial_num']=='s7'].index.values)
    subclass_test = np.concatenate([subclass_test8, class_1_test2, class_1_test3, class_3_test2, class_3_test3], axis=0)
    testing = np.array(testing.index.values)
    
    if use_1517:
        logger.info("15/17 Split")
        return training, testing
    else:
        logger.info("Seen/Unseen Split")
        return subclass, subclass_test

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
    model = model_.ManifoldNetRes().cuda()
    logger.info(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("#Model Parameters: "+str(params))
    
    batches = [50, 100, 200, 400, 800]
    lrs = [0.01, 0.015, 0.03, 0.05, 0.005, 0.08, 0.1, 0.005, 0.001]
    
    
    for i in batches:
        train_loader, test_loader = data_prep(args.data_dir, i, args.test_batchsize, use_1517, logger)
        logger.info("Batch Size: "+str(args.batchsize))
        for j in lrs:
            optimizer = optim.Adam(model.parameters(), lr=j, eps=1e-8, amsgrad=True)
            logger.info("Learning Rate: "+str(args.lr))
            for epoch in range(1, args.epochs + 1):
                
                acc=test(model, device, test_loader, logger, epoch)
                if acc > highest:
                    if save_path is not None:
                        try:
                            os.remove(save_path+'.ckpt')
                        except:
                            pass
                    highest = acc
                    save_path = os.path.join('./save/', '[{acc}]-[{batch}]-[{learning_rate}]-11class-model-[{model}]'.format(acc = np.round(acc, 3), batch=i, learning_rate=j, model=use_1517))
                    torch.save(model.state_dict(), save_path+'.ckpt')
                    logger.info('Saved model checkpoints into {}...'.format(save_path))
                
                train(model, device, train_loader, optimizer, epoch, logger)
                
                
            logger.info("########## NEW MODEL ###########")
            model = model_.ManifoldNetRes().cuda()

if __name__ == '__main__':
    main()
       

