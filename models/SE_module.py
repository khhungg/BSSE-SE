import torch, importlib, fairseq
import torch.nn as nn
# from unilm.wavlm.WavLM import WavLM, WavLMConfig 
from models.WavLM import WavLM, WavLMConfig 
from util import get_feature, feature_to_wav

ssl_model_path = {
    'wavlm_base'    :'./save_model/WavLM-Base+.pt',
    'wavlm_large'   :'./save_model/WavLM-Large.pt',
    'wav2vec2_base' :'./save_model/wav2vec_small.pt',
    'wav2vec2_large':'./save_model/libri960_big.pt',
    'hubert_base'   :'./save_model/hubert_base_ls960.pt',
    'hubert_large'  :'./save_model/hubert_large_ll60k.pt'
}

class SE_module(nn.Module):
    
    def __init__(self,args):
        super().__init__()
        if not args.feature=='raw':
            ssl_path = ssl_model_path[f'{args.ssl_model}_{args.size}']
            if args.ssl_model=='wavlm':
                checkpoint = torch.load(ssl_path)
                cfg = WavLMConfig(checkpoint['cfg'])
                cfg.encoder_layerdrop = 0
                self.model_SSL = WavLM(cfg)
                self.model_SSL.load_state_dict(checkpoint['model'])
            else:
                model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task( [ssl_path],arg_overrides={'encoder_layerdrop':0})
                self.model_SSL = model[0]
                self.model_SSL.remove_pretraining_modules()
        self.args = args  
        model_SE  =  importlib.import_module('models.'+args.model).__getattribute__('MainModel')
        self.model_SE = model_SE(args)
        self.model_SE.apply(self.weights_init)
        
    def weights_init(self,m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
                    
    def forward(self,wav,output_wav=False,layer_norm=True):
        

#         generate SSL feature
        if self.args.feature!='raw':
            if self.args.ssl_model=='wavlm':
                rep, layer_results = self.model_SSL(wav, output_layer=self.model_SSL.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
            else:
                layer_results = self.model_SSL(wav, mask=False, features_only=True)['layer_results']
                layer_reps = [x.transpose(0, 1) for x, _, _ in layer_results]
        else:
            layer_reps = None
            
        out = self.model_SE(wav,layer_reps=layer_reps,output_wav=output_wav,layer_norm=layer_norm)

        return out