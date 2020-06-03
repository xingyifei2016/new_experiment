from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import random, sys
import _pickle as cPickle
import math
import numpy as np
import layers
from pdb import set_trace as st
from logger import setup_logger
import shrinkage_layers


def calc_next(inputs, kern, stride, outs):
    
    if type(kern) == int:
        dims = int(math.floor((inputs-(kern-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, dims])
    else:
        dims = int(math.floor((inputs-(kern[0]-1)-1)/stride + 1))
        return torch.Size([2, outs, dims, 1])
    
    
    
class ManifoldNetComplex(nn.Module):
    def __init__(self):
        super(ManifoldNetComplex, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 1), (5, 1), num_tied_block=1)
        self.proj1 = layers.manifoldReLUv2angle(20) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_1 = layers.DistanceTransform(20, (2, 1), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 40, (6, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (3, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.name = "Complex"

    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x = self.linear_1((x))
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

class ManifoldNetComplex(nn.Module):
    def __init__(self):
        super(ManifoldNetComplex, self).__init__()
        self.complex_conv1 = layers.ComplexConv(1, 20, (5, 1), (2, 1), num_tied_block=1)
        self.proj1 = layers.manifoldReLUv2angle(20) 
        self.proj2 = layers.manifoldReLUv2angle(20) 
        self.complex_conv2 = layers.ComplexConv(20, 20, (5, 1), (2, 1), num_tied_block=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_1 = layers.DistanceTransform(20, (2, 1), num_tied_block=1)
        self.conv_1 = nn.Conv2d(20, 40, (6, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (4, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (4, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.name = "Complex"

    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x = self.complex_conv2(x)
        x = self.proj2(x)
        x = self.linear_1((x))
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
    
    
class ManifoldNetComplex(nn.Module):
    def __init__(self, num_distr=5, ws=1):
        super(ManifoldNetComplex, self).__init__()
        self.complex_conv1 = shrinkage_layers.ComplexConv2Deffangle4Dxy(1, 20, (5, 1), (5, 1))
        self.proj1 =  shrinkage_layers.ReLU4Dsp(20)
        params={'num_classes': 11, 'num_distr': num_distr}
        self.SURE =  shrinkage_layers.SURE_pure4D(params, calc_next(128, (5, 1), 5, 20), 20, weight=ws)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.conv_1 = nn.Conv2d(20, 40, (5, 1))
        self.mp_1 = nn.MaxPool2d((2,1))
        self.conv_2 = nn.Conv2d(40, 60, (5, 1))
        self.mp_2 = nn.MaxPool2d((2,1))
        self.conv_3 = nn.Conv2d(60, 80, (3, 1))
        self.bn_1 = nn.BatchNorm2d(40)
        self.bn_2 = nn.BatchNorm2d(60)
        self.bn_3 = nn.BatchNorm2d(80)
        self.linear_2 = nn.Linear(80, 40)
        self.linear_3 = nn.Linear(40, 11)
        self.loss_weight = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.name = "Regular Network"
    def forward(self, x, labels=None):
        x = self.complex_conv1(x)
        x = self.proj1(x)
        x, losses = self.SURE(x, labels)
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
        return x, res_loss


def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data, target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target) + torch.sum(_)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
            
def train(args, model, device, train_loader, optimizer, epoch, logger):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data, target)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target) + torch.sum(_[0])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
        st()
            
            
def eval_train(args, model, device, test_loader, logger):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = []
    real_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
             
    test_loss /= len(test_loader.dataset)
    logger.info('\nTraining set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))           
            
def test(args, model, device, test_loader, lbl, snrs, test_idx, logger):
    model.eval()
    test_loss = 0
    correct = 0
    pred_all = []
    real_all = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            pred_all.append(np.array(pred.cpu()))
            real_all.append(np.array(target.cpu()))
            correct += pred.eq(target.view_as(pred)).sum().item()
    pred_all = np.squeeze( np.concatenate(pred_all) )
    real_all = np.concatenate(real_all)
    
    acc = {}
    for snr in snrs:

        # extract classes @ SNR
        test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
        pred_i = pred_all[np.where(np.array(test_SNRs)==snr)]
        real_i = real_all[np.where(np.array(test_SNRs)==snr)]
        logger.info('SNR ' +str(snr)+' test accuracy: '+str(100. * np.mean(pred_i==real_i) )+'%.')
             
    test_loss /= len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct
    


def to_polar4D(X):
    M = np.linalg.norm(X, axis=1)
    T = np.arctan2(X[:,1,:],X[:,0,:])
    MT = np.amax(T)
    mT = np.amin(T)
    MM = np.amax(M)
    mM = np.amin(M)
    
    # [log(|z|), x/|z|, y/|z|]
    
    # log|z|
    log_mag = np.log(np.expand_dims(M, axis=1))
    
    # x/|z|, y/|z|
    X_transformed = X / np.expand_dims(M, axis=1)
    
    Y = np.concatenate((log_mag, X),axis=1)
    
    Y = np.expand_dims(np.expand_dims(Y,axis=3),axis=2)
    
    return Y


def to_polar4D(X):
    M = np.linalg.norm(X, axis=1)
    T = np.arctan2(X[:,1,:],X[:,0,:])
    MT = np.amax(T)
    mT = np.amin(T)
    MM = np.amax(M)
    mM = np.amin(M)
    Y = np.expand_dims(np.expand_dims(np.concatenate((np.expand_dims(T,axis=1),np.expand_dims(M,axis=1),X),axis=1),axis=3),axis=2)
    return Y


def data_prep(path, train_idx_path, test_idx_path, batch_size_train, batch_size_test):
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(list(yy))),yy] = 1
        return yy1
    
    Xd = cPickle.load(open(path,'rb'), encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    X_train = X[train_idx]
    X_test = X[test_idx]
    X_train = (X_train - np.mean(X_train) ) / np.std(X_train)
    X_test = (X_test - np.mean(X_test) ) / np.std(X_test)
    X_train = to_polar4D(X_train)
    X_test = to_polar4D(X_test)
    Y_train = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), train_idx)) )
    Y_test = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), test_idx)) )
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy (Y_train).type(torch.LongTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size_train, shuffle = True)
    test_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size_test, shuffle = False)
    return train_loader_dataset, test_loader_dataset, lbl, snrs, test_idx

def main():
    #argparse settings
    parser = argparse.ArgumentParser(description='PyTorch RadioML Example') #400 and 0.001
    parser.add_argument('--batchsize', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 400)')
    parser.add_argument('--test_batchsize', type=int, default=400, metavar='N',
                        help='input batch size for testing (default: 400)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.03, metavar='LR',
                        help='learning rate (default: 0.03)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Adam momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data-dir', type=str, default="/home/yifei/complex_demo/data/RML2016.10a_dict.pkl", metavar='N',
                        help='where data is stored')
    parser.add_argument('--train-id', type=str, default="/home/yifei/complex_demo/data/train_idx.npy", metavar='N',
                        help='where train ids are stored')
    parser.add_argument('--test-id', type=str, default="/home/yifei/complex_demo/data/test_idx.npy", metavar='N',
                        help='where test ids are stored')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    model = ManifoldNetComplex().to(device)
#     model.load_state_dict(torch.load(save_path))
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#Model Parameters: "+str(params))
    train_loader, test_loader, lbl, snrs, test_idx = data_prep(args.data_dir, args.train_id, args.test_id, args.batchsize, args.test_batchsize)
    print("Batch Size: "+str(args.batchsize))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, amsgrad=True)
    print("Learning Rate: "+str(args.lr))
    
    
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
    logger.info(model)
    batches = [200, 400, 800]
    lrs = [0.03, 0.05, 0.005, 0.08, 0.1, 0.005, 0.001]
    
    batches = [100]
    lrs = [0.02]
    
    ws = [0, 0.5, 1, 5, 10]
            
    for w in ws:        
        model = ManifoldNetComplex(5, w).cuda()
        
        logger.info(w)
        
        for i in batches:
            train_loader, test_loader, lbl, snrs, test_idx = data_prep(args.data_dir, args.train_id, args.test_id, args.batchsize, args.test_batchsize)
            logger.info("Batch Size: "+str(args.batchsize))
            for j in lrs:
                optimizer = optim.Adam(model.parameters(), lr=j, eps=1e-8, amsgrad=True)
                logger.info("Learning Rate: "+str(args.lr))
                for epoch in range(1, args.epochs + 1):

                    acc=test(args, model, device, test_loader, lbl, snrs, test_idx, logger)
                    if acc > highest:
                        if save_path is not None:
                            try:
                                os.remove(save_path+'.ckpt')
                            except:
                                pass
                        highest = acc
                        save_path = os.path.join('./save/', '[{acc}]-[{batch}]-[{learning_rate}]-11class-model'.format(acc = np.round(acc, 3), batch=i, learning_rate=j))
                        torch.save(model.state_dict(), save_path+'.ckpt')
                        logger.info('Saved model checkpoints into {}...'.format(save_path))

                    train(args, model, device, train_loader, optimizer, epoch, logger)


                logger.info("########## NEW MODEL ###########")
                model = ManifoldNetComplex().cuda()
            
    

if __name__ == '__main__':
    main()
