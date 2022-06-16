import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import timeit
import sys
import pdb
from conv_stft import ConvSTFT, ConviSTFT
#from preprocess import TorchOLA
from models.dual_transf import Dual_Transformer

class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class DPTNet(nn.Module):
    def __init__(self, args, L=400, width=64):
        super(DPTNet, self).__init__()
        self.L = L
        self.frame_shift = 160
        self.args = args
        if not args.feature=='raw':
            self.dim = 768 if 'base' in args.size else 1024
            weight_dim = 13 if 'base' in args.size else 25
            if not args.ssl_model=='wavlm':
                weight_dim = weight_dim-1
            if args.weighted_sum:
                self.weights = nn.Parameter(torch.ones(weight_dim))
                self.softmax = nn.Softmax(-1)
                layer_norm  = []
                for _ in range(weight_dim):
                    layer_norm.append(nn.LayerNorm(self.dim))
                self.layer_norm = nn.Sequential(*layer_norm)
            
        if args.feature=='raw':
            embed = 200
        elif args.feature=='ssl':
            embed = self.dim
        else:
            embed = 200+self.dim
            
#         self.N = 256
#         self.B = 256
#         self.H = 512
#         self.P = 3
        # self.device = device
        self.in_channels = 1 if args.feature=='ssl' else 2
        self.in_linear   = nn.Linear(embed, 200, bias=True)
        self.out_channels = 2
        self.kernel_size = (2, 3)
        self.fea_size = L//2+1
        # self.elu = nn.SELU(inplace=True)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.width = width

        self.inp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.width, kernel_size=(1, 1))  # [b, 64, nframes, 256]
        self.inp_norm = nn.LayerNorm(self.fea_size-1)
        self.inp_prelu = nn.PReLU(self.width)

#         self.enc_dense1 = DenseBlock(self.fea_size-1, 4, self.width)
        self.enc_dense1 = DenseBlock(self.fea_size-1, 3, self.width)
        
        '''
        self.enc_conv1 = nn.Conv2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), stride=(1, 1))  # [b, 64, nframes, 256]
        self.enc_norm1 = nn.LayerNorm(256)
        self.enc_prelu1 = nn.PReLU(self.width)
        '''

#         self.dual_transformer = Dual_Transformer(64, 64, num_layers=4)  # # [b, 64, nframes, 8]
        self.dual_transformer = Dual_Transformer(64, 64, num_layers=3)  # # [b, 64, nframes, 8]
        
        # gated output layer
        self.output1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Tanh()
        )
        
        self.output2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Sigmoid()
        )

#         self.dec_dense1 = DenseBlock(self.fea_size-1, 4, self.width)
        self.dec_dense1 = DenseBlock(self.fea_size-1, 3, self.width)
        
        '''
        self.dec_conv1 = SPConvTranspose2d(in_channels=self.width, out_channels=self.width, kernel_size=(1, 3), r=1)
        self.dec_norm1 = nn.LayerNorm(256)
        self.dec_prelu1 = nn.PReLU(self.width)
        '''


        self.out_conv = nn.Conv2d(in_channels=self.width, out_channels=self.out_channels, kernel_size=(1, 1))
       # self.ola = ISTFT(self.frame_shift)#TorchOLA(self.frame_shift)
        self.stft =  ConvSTFT(self.L, self.frame_shift, self.L, 'hanning', 'real', fix=True)
        self.istft = ConviSTFT(self.L, self.frame_shift, self.L, 'hanning', 'complex', fix=True)

        #show_model(self)
        #show_params(self)

    def forward(self, x, layer_reps, output_wav=True,layer_norm=True, masking_mode='C'):

        #masking_mode = 'R'
       # print(x.shape)
       # x = torch.squeeze(x,1)
        length = x.shape[-1]
        x = F.pad(x.unsqueeze(1),(0,self.frame_shift-x.shape[-1]%self.frame_shift),"constant",0)
        x = self.stft(x)
        #real = x[:,:257,:]
        #imag = x[:,257:,:]
        real = x[0]
        imag = x[1]
        #pdb.set_trace()
        x = torch.stack([real,imag],1)
        x = x.permute(0,1,3,2) # [B, 2, num_frames, num_bins]
        x = x[...,1:]
        real = x[:,0]
        imag = x[:,1]
        #print(x.shape)
        
        if self.args.feature!='raw':
            if self.args.weighted_sum:
                ssl = torch.cat(layer_reps,2)
            else:
                ssl = layer_reps[-1]
            B,T,embed_dim = ssl.shape
            ssl = ssl.repeat(1,1,2).reshape(B,-1,embed_dim)
            ssl = torch.cat((ssl,ssl[:,-1:].repeat(1,x.shape[2]-ssl.shape[1],1)),dim=1)
            if self.args.weighted_sum:
                lms  = torch.split(ssl, self.dim, dim=2)
                for i,(lm,layer,weight) in enumerate(zip(lms,self.layer_norm,self.softmax(self.weights))):
                    if layer_norm:
                        lm = layer(lm)
                    if i==0:
                        out = lm*weight
                    else:
                        out = out+lm*weight
#                 pdb.set_trace()
                x    = torch.cat((x,out.unsqueeze(1).repeat(1,2,1,1)),-1) if self.args.feature=='cross' else out.unsqueeze(1) 
            else:
#                 pdb.set_trace()
                x    = torch.cat((x,ssl.unsqueeze(1).repeat(1,2,1,1)),-1) if self.args.feature=='cross' else ssl.unsqueeze(1)
            
        x = self.in_linear(x)
        

        out = self.inp_prelu(self.inp_norm(self.inp_conv(x)))  # [b, 64, num_frames, frame_size]

        x1 = self.enc_dense1(out)   # [b, 64, num_frames, frame_size]
        #x1 = self.enc_prelu1(self.enc_norm1(self.enc_conv1(self.pad1(out))))  # [b, 64, num_frames, 256]
        #print(x1.shape)  
        out = self.dual_transformer(x1)  # [b, 64, num_frames, 256]


        out = self.output1(out) * self.output2(out)  # mask [b, 64, num_frames, 256]
        #out = x1 * out

        out = self.dec_dense1(out)
        #out = self.dec_prelu1(self.dec_norm1(self.dec_conv1(self.pad1(out))))

        out = self.out_conv(out)
        #out = self.ola(out)
       # print(out.shape)
#        out = out.permute(0,3,2,1)
#        out = torch.cat((torch.zeros((out.size()[0], 1, out.size()[2], 2)).to(device='cuda'), out), 1)


        mask_real = out[:,0]
        mask_imag = out[:,1]
                                                                            

        if masking_mode == 'E' :
            mask_mags = (mask_real**2+mask_imag**2)**0.5
            real_phase = mask_real/(mask_mags+1e-8)
            imag_phase = mask_imag/(mask_mags+1e-8)
            mask_phase = torch.atan2( imag_phase, real_phase )
            #mask_mags = torch.clamp_(mask_mags,0,100) 
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags*spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags*torch.cos(est_phase)
            imag = est_mags*torch.sin(est_phase)
        elif masking_mode == 'C':
            real,imag = real*mask_real-imag*mask_imag, real*mask_imag+imag*mask_real
        elif masking_mode == 'R':
            real, imag = real*mask_real, imag*mask_imag

        #print(out.shape)
        #pdb.set_trace()
        real = torch.cat((torch.zeros((real.size()[0], real.size()[1], 1)).to(x.device), real), -1)
        imag = torch.cat((torch.zeros((imag.size()[0], imag.size()[1], 1)).to(x.device), imag), -1)
        out = torch.cat([real,imag],-1).permute(0,2,1)

       # print(out.shape)
        out = self.istft(out).squeeze(1)
        out = torch.clamp_(out,-1,1)
        #out = torch.istft(out, n_fft=self.L, hop_length=self.frame_shift)

        return out[...,:length]
    
    
def MainModel(args):
    
    model = DPTNet(args)
    
    return model

'''
x = torch.ones((2, 1, 250, 512))
model = Net()
out = model(x)
print(out.shape)


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


print(numParams(model))
'''
'''
x = torch.ones((2, 1, 250, 512))
model = Net()
out = model(x)
print(out.shape)


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


print(numParams(model))
'''
