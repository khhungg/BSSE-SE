import torch.nn as nn
import torch, time, torchaudio
import pandas as pd
import os, sys
from tqdm import tqdm
import pdb
import numpy as np
from util import get_filepaths, check_folder, cal_score, get_feature, progress_bar
import torch.fft as fft
from pesq import pesq
from pystoi.stoi import stoi
from torch.optim import Adam
    
class Trainer:
    def __init__(self, model, epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader,Test_path, writer, model_path, score_path, args):
#         self.step = 0
        self.epoch = epoch
        self.epochs = epochs
        self.best_loss = best_loss
        self.model = model.to(device)
        self.optimizer = optimizer
        self.fea   = args.feature

        self.device = device
        self.loader = loader
        self.criterion = criterion
        self.Test_path = Test_path

        self.train_loss = 0
        self.val_loss = 0
        self.writer = writer
        self.model_path = model_path
        self.score_path = score_path
        self.args = args
        self.transform = get_feature()
        if not args.finetune_SSL and args.feature!='log1p':
            for name,param in self.model.model_SSL.named_parameters():
                param.requires_grad = False         
        if args.finetune_SSL=='PF':
            for name,param in self.model.model_SSL.feature_extractor.named_parameters():
                param.requires_grad = False            

    def save_checkpoint(self,):
        save_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state_dict = {
            'epoch': self.epoch,
            'model': save_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss
            }
        check_folder(self.model_path)
        torch.save(state_dict, self.model_path)
    
    def get_fea(self,wav,ftype='log1p'):
        if ftype=='wav':
            return wav
        elif ftype=='complex':
            return self.transform(wav,ftype=ftype)[0]
        else:
            return self.transform(wav,ftype=ftype)[0][0]
        
    def _train_step(self, nwav,cwav):
        device = self.device
        nwav,cwav = nwav.to(device),cwav.to(device)
        cdata = self.get_fea(cwav)
        pred = self.model(nwav)
        loss = self.criterion(pred,cdata)
        self.train_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


#             if USE_GRAD_NORM:
#                 nn.utils.clip_grad_norm_(self.model['discriminator'].parameters(), DISCRIMINATOR_GRAD_NORM)
#             self.optimizer['discriminator'].step()


    def _train_epoch(self):
        self.train_loss = 0
        self.model.train()
        step = 0
        t_start =  time.time()
        for nwav,cwav in self.loader['train']:
            self._train_step(nwav,cwav)
            step += 1
            progress_bar(self.epoch,self.epochs,step,self.train_step,time.time()-t_start,loss=self.train_loss,mode='train')
        self.train_loss /= len(self.loader['train'])
        print(f'train_loss:{self.train_loss}')

    
#     @torch.no_grad()
    def _val_step(self, nwav,cwav):
        device = self.device
        nwav,cwav = nwav.to(device),cwav.to(device)
        cdata = self.get_fea(cwav)
        pred = self.model(nwav)
        loss = self.criterion(pred,cdata)
        self.val_loss += loss.item()


    def _val_epoch(self):
        self.val_loss = 0
        self.model.eval()
        step = 0
        t_start =  time.time()
        for nwav,cwav in self.loader['val']:
            self._val_step(nwav,cwav)
            step += 1
            progress_bar(self.epoch,self.epochs,step,self.val_step,time.time()-t_start,loss=self.val_loss,mode='test')
        self.val_loss /= len(self.loader['val'])
        print(f'val_loss:{self.val_loss}')
        
        if self.best_loss > self.val_loss:
            
            print(f"Save model to '{self.model_path}'")
            self.save_checkpoint()
            self.best_loss = self.val_loss
            
    def write_score(self,test_file,c_path):
        args = self.args
        wavname = test_file.split('/')[-1]
        c_file  = os.path.join(c_path,wavname)
        n_data,sr = torchaudio.load(test_file)
        c_data,sr = torchaudio.load(c_file)

        enhanced  = self.model(n_data.to(self.device),output_wav=True)
        out_path = f'./Enhanced/{self.model.__class__.__name__}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_{args.feature}_{args.size}_'\
                    f'WS{args.weighted_sum}_FT{args.finetune_SSL}/{wavname}'
                    
        check_folder(out_path)
        enhanced = enhanced.cpu()
        torchaudio.save(out_path,enhanced,sr)
            
        s_pesq, s_stoi, s_snr, s_sdr = cal_score(c_data.squeeze().detach().numpy(),enhanced.squeeze().detach().numpy())
        with open(self.score_path, 'a') as f:
            f.write(f'{wavname},{s_pesq},{s_stoi},{s_snr},{s_sdr}\n')
            
    

    def train(self):
        args = self.args
        model_name = self.model.module.__class__.__name__ if isinstance(self.model, nn.DataParallel) else self.model.__class__.__name__        
        figname = f'{self.args.task}/{model_name}_{args.target}_{args.feature}_{args.size}_WS{args.weighted_sum}_FT{args.finetune_SSL}'
        self.train_step = len(self.loader['train'])
        self.val_step = len(self.loader['val'])
        while self.epoch < self.epochs:
            self._train_epoch()
            self._val_epoch()

            self.writer.add_scalars(f'{figname}', {'train': self.train_loss},self.epoch)
            self.writer.add_scalars(f'{figname}', {'val': self.val_loss},self.epoch)
            self.epoch += 1
            
    def test(self):
        # load model
        self.model.eval()
        checkpoint      = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint['model'])
        noisy_paths        = get_filepaths(self.Test_path['noisy'])
        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)
        with open(self.score_path, 'a') as f:
            f.write('Filename,PESQ,STOI,SISNR,SDR\n')
        for noisy_path in tqdm(noisy_paths):
            self.write_score(noisy_path,self.Test_path['clean'])

        data = pd.read_csv(self.score_path)
        pesq_mean = data['PESQ'].to_numpy().astype('float').mean()
        stoi_mean = data['STOI'].to_numpy().astype('float').mean()
        snr_mean  = data['SISNR'].to_numpy().astype('float').mean()
        sdr_mean  = data['SDR'].to_numpy().astype('float').mean()

        with open(self.score_path, 'a') as f:
            f.write(','.join(('Average',str(pesq_mean),str(stoi_mean),str(snr_mean),str(sdr_mean)))+'\n')