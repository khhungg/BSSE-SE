import os, argparse, torch, random, sys, torchaudio
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import pandas as pd
import pdb
from models.SE_module import SE_module
# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

torchaudio.set_audio_backend("sox_io")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_folder', type=str, default='../noisy-vctk-16k')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--feature', type=str, default='raw') # raw / ssl / cross
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='BLSTM')    
    parser.add_argument('--ssl_model', type=str, default=None) # wav2vec2 / hubert / wavlm
    parser.add_argument('--size', type=str, default='base')  # base / large
    parser.add_argument('--finetune_SSL', type=str, default=None) # PF / EF
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--target', type=str, default='MAP') #'MAP' or 'IRM'
    parser.add_argument('--task', type=str, default='SSL_SE') 
    parser.add_argument('--weighted_sum' , action='store_true')    
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    return args

def get_path(args):
    # data path
    Train_path = {
        'noisy':f'{args.data_folder}/noisy_trainset_28spk_wav_16k/',
        'clean':f'{args.data_folder}/clean_trainset_28spk_wav_16k/',
        }
    Test_path = {
        'noisy':f'{args.data_folder}/noisy_testset_wav_16k/',
        'clean':f'{args.data_folder}/clean_testset_wav_16k/',
        }
    checkpoint_path = f'./checkpoint/{args.model}_{args.ssl_model}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_{args.feature}_{args.size}_'\
                    f'WS{args.weighted_sum}_FT{args.finetune_SSL}.pth.tar'
    model_path = f'./save_model/{args.model}_{args.ssl_model}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_batch{args.batch_size}_'\
                    f'lr{args.lr}_{args.feature}_{args.size}_'\
                    f'WS{args.weighted_sum}_FT{args.finetune_SSL}.pth.tar'
    score_path = f'./Result/{args.model}_{args.ssl_model}_{args.target}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr}_{args.feature}_{args.size}_'\
                    f'WS{args.weighted_sum}_FT{args.finetune_SSL}.csv'
    
    return Train_path,Test_path,checkpoint_path,model_path,score_path


if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    print(cwd)
    print(SEED)
    torch.backends.cuda.cufft_plan_cache.max_size = 0
    
    
    # get parameter
    args = get_args()
    
    # declair path
    Train_path,Test_path,checkpoint_path,model_path,score_path = get_path(args)
    
    # tensorboard
    writer = SummaryWriter('/home/khhung/logs')

#     exec (f"from models.{args.model.split('_')[0]} import {args.model} as model")
#     model     = model(args)
    model     = SE_module(args)
    model, epoch, best_loss, optimizer, criterion, device = Load_model(args,model,checkpoint_path, model_path)
    loader = Load_data(args, Train_path)        
    Trainer = Trainer(model, args.epochs, epoch, best_loss, optimizer, 
                      criterion, device, loader, Test_path, writer, model_path, score_path, args)
    try:
        if args.mode == 'train':
            Trainer.train()
        Trainer.test()
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
