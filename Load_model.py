import torch, os, numpy as np, random, torchaudio
import torch.nn as nn
from torch.optim import Adam
from util import get_filepaths
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
from tqdm import tqdm 
from loss import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Load_model(args,model,checkpoint_path,model_path):
    
    criterion = {
        'mse'     : nn.MSELoss(),
        'l1'      : nn.L1Loss(),
        'DPTLoss' : DPTLoss()
    }
    device    = torch.device(f'cuda:{args.gpu}')
    criterion = criterion[args.loss_fn].to(device)
    if args.feature!='log1p':
        optimizers = {
            'adam'    : Adam([
                {'params': model.model_SE.parameters()},
                {'params': model.model_SSL.parameters(), 'lr': args.lr*0.1}
            ],lr=args.lr,weight_decay=0)}
    else:
        optimizers = {
            'adam'    : Adam(model.parameters(),lr=args.lr,weight_decay=0)}
    optimizer = optimizers[args.optim]
    epoch = 0
    best_loss = 10
    para = count_parameters(model)
    print(f'Num of model parameter : {para}')
    return model,epoch,best_loss,optimizer,criterion,device


def Load_data(args, Train_path):

    file_paths             = get_filepaths(Train_path['noisy'])
    train_paths, val_paths = train_test_split(file_paths,test_size=0.05,random_state=13162)
    train_dataset          = CustomDataset(args, train_paths, Train_path['clean'])
    val_dataset            = CustomDataset(args, val_paths, Train_path['clean'], val=True)
    loader = { 
        'train':DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True),
        'val'  :DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    }
    return loader

class CustomDataset(Dataset):

    def __init__(self, args, paths, cpath, length=128, hop_length=160, val=False):   # initial logic happens like transform
        
        self.args          = args
        self.paths         = paths
        self.cpath         = cpath
        self.max_audio     = length*hop_length
        self.val           = val
        self.weighted_sum  = args.weighted_sum

    def load_audio(self,path,):

        wav    = torchaudio.load(path)[0]
        cpath  = os.path.join(self.cpath,path.split('/')[-1])
        cwav   = torchaudio.load(cpath)[0]
        frame  = wav.shape[1]//160+1
        
        if wav.shape[0]!=1:
            i = torch.randperm(wav.shape[0])[0]
            wav  = wav[i:i+1]
            cwav = cwav[i:i+1]
        
        if wav.shape[-1]<=self.max_audio:
            time = self.max_audio//wav.shape[-1]+1
            wav  = wav.repeat(1,time)
            cwav = cwav.repeat(1,time)

        start   = torch.randint(0,(wav.shape[1]-self.max_audio),(1,))
        n_audio = wav[0,start : start + self.max_audio]
        c_audio = cwav[0,start : start + self.max_audio]

        if self.val:
            return n_audio, c_audio
        else:
            scale = torch.rand(1)*1.5+1
            return n_audio*scale, c_audio*scale

    def __getitem__(self, index):
        npath = self.paths[index]
        n_wav, c_wav = self.load_audio(npath)
        
        return n_wav, c_wav

    def __len__(self):  # return count of sample we have
        
        return len(self.paths)


