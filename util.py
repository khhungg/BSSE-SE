#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch, os, torchaudio, math, numpy as np, sys
import torch.nn.functional as F
from torchaudio import functional as taF
import os, pdb
# from pypesq import pesq
from pesq import pesq
from pystoi.stoi import stoi
from mir_eval.separation import bss_eval_sources
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

n_fft      = 400
win_length = 400
hop_length = 160
n_mels     = 40
n_mfcc     = 40
epsilon    = 1e-6
top_DB     = 80
class get_feature():
    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr
        self.torchmscale=torchaudio.transforms.MelScale(
            n_stft=n_fft//2+1, 
            n_mels=n_mels, 
            sample_rate=sr)
        self.dct_mat = taF.create_dct(n_mfcc, n_mels, norm='ortho')
        self.amplitude_to_DB = torchaudio.transforms.AmplitudeToDB('power', top_DB)
        
    def __call__(self, wav, sr=16000, ftype='log1p', log=False, norm=None):
        self.torchmscale = self.torchmscale.to(wav.device)
        self.dct_mat     = self.dct_mat.to(wav.device)
        self.amplitude_to_DB = self.amplitude_to_DB.to(wav.device)

        length = wav.shape[-1]
        phase_list = ['spec','log1p']
        x_stft = torch.stft(
                wav, 
                n_fft=n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                center=True, 
                normalized=False, 
                onesided=True,
                pad_mode='reflect',
                return_complex=False,
                window=torch.hamming_window(win_length).to(wav.device))
#         print(x_stft.shape)
        if ftype=='complex':
            feature = x_stft
        elif ftype=='spec' or ftype=='log1p':
            feature  = torch.norm(x_stft,dim=-1,p=2)+epsilon
            phase    = x_stft/feature.unsqueeze(-1)
            if ftype=='log1p': 
                feature = torch.log1p(feature)
        elif ftype=='mel_spec' or ftype=='mfcc':
            feature = self.torchmscale(torch.view_as_complex(x_stft).abs().pow(2))
            if ftype=='mfcc':
                feature = self.amplitude_to_DB(feature)
                feature = torch.matmul(feature.transpose(-2, -1), self.dct_mat).transpose(-2, -1)
        if log:feature = (feature).log()
        if ftype in phase_list:return (feature,phase), length
        else:return feature, length
    
    
def feature_to_wav(fea, length, ftype='log1p', log=False):
#     ftype : complex, spec, log1p
    phase_list = ['spec','log1p']
    if ftype in phase_list:
        fea, phase = fea
    device = fea.device
    if log:
        fea = torch.exp(fea)
    if ftype=='log1p':
        fea = torch.expm1(fea)
    if ftype in phase_list:
        fea = phase*(fea-epsilon).unsqueeze(-1)
        
    wav = torch.istft(
            fea, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            center=True, 
            normalized=False, 
#             onesided=True, 
            window=torch.hamming_window(win_length).to(device),
            return_complex=False,
            length=length
        )
    return wav

def check_path(path):
    if not os.path.isdir(path): 
        os.makedirs(path)
        
def check_folder(path):
    path_n = '/'.join(path.split('/')[:-1])
    check_path(path_n)
    
def get_filepaths(directory,ftype='.wav'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.

    return sorted(file_paths)

def cal_score(clean,enhanced):
#     if not clean.shape==enhanced.shape:
#         enhanced = enhanced[:clean.shape]
#     clean = clean/abs(clean).max()
#     enhanced = enhanced/abs(enhanced).max()
    try:
        s_stoi = stoi(clean, enhanced, 16000)
    except:
        s_stoi = 0 
#     s_pesq = pesq(clean, enhanced, 16000)
    s_pesq = pesq(16000, clean, enhanced, 'wb')
    s_snr  = si_snr(enhanced,clean)
    s_sdr  = bss_eval_sources(clean,enhanced,False)[0][0]
    if math.isnan(s_pesq):
        s_pesq=0
    if math.isnan(s_stoi):
        s_stoi=0
    return round(s_pesq,5), round(s_stoi,5), round(s_snr,5), round(s_sdr,5)

def si_snr(x, s, remove_dc=True):

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))

def progress_bar(epoch, epochs, step, n_step, time, loss, mode):
    line = []
    line = f'\rEpoch {epoch}/ {epochs}'
    loss = round(loss/step,4)
    if step==n_step:
        progress = '='*10
    else :
        n = int(10*step/n_step)
        progress = '='*n + '>' + '.'*(9-n)
    eta = time*(n_step-step)/step
    def sec_to_time(time):
        mm, ss = divmod(time,60)
        hh, mm = divmod(mm,60)
        return hh, mm ,ss
    ihh, imm, iss = sec_to_time(int(time))
    lhh, lmm, lss = sec_to_time(int(eta))
    line += f'[{progress}] - {step}/{n_step} |{ihh:02.0f}:{imm:02.0f}:{iss:02.0f} |{lhh:02.0f}:{lmm:02.0f}:{lss:02.0f}|{mode}:{loss}'
    if step==n_step:
        line += '\n'
    sys.stdout.write(line)
    sys.stdout.flush()