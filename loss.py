import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy
from conv_stft import ConvSTFT, ConviSTFT


class DPTLoss(nn.Module):
    def __init__(self,):
        super(DPTLoss, self).__init__()
        
        self.L2loss  = nn.MSELoss()
        self.L1loss  = nn.L1Loss()
        self.nn_stft = ConvSTFT(512, 256, 512, 'hanning', 'real', fix=True)

    def forward(self, pred, cdata):
        
        loss_wav = self.L2loss(pred, cdata)
        ecomp = self.nn_stft(pred)
        e_real = ecomp[0]
        e_img = ecomp[1]
        ccomp = self.nn_stft(cdata.cuda())
        c_real = ccomp[0]
        c_img = ccomp[1]
        loss_spec = self.L1loss(e_real, c_real) + self.L1loss(e_img, c_img) 
        loss = loss_wav*0.5+loss_spec
        
        return loss

