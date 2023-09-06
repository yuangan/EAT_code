import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from modules.audioencoder import MappingNetwork, AudioEncoder, MappingDeepNetwork

class EmotionPrompt(nn.Module):
    def __init__(self):
        super(EmotionPrompt, self).__init__()
        self.mappingnet = MappingNetwork(latent_dim=16, style_dim=128, num_domains=8, hidden_dim=512)
    
    def forward(self,x):

        z_trg = x['z_trg']
        y_org = x['y_trg']
        s_trg = self.mappingnet(z_trg, y_org)

        return s_trg

# num_domain = 1
class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.mappingnet = MappingDeepNetwork(latent_dim=16, style_dim=128, num_domains=1, hidden_dim=512)
    
    def forward(self,x):

        z_trg = x['z_trg']
        y_org = x['y_trg']
        bs = y_org.shape[0]
        s_trg = self.mappingnet(z_trg, y_org).reshape(bs, -1, 128)

        return s_trg[:,0,:], s_trg[:,1:,:]

class EmotionDeepPrompt(nn.Module):
    def __init__(self):
        super(EmotionDeepPrompt, self).__init__()
        self.mappingnet = MappingDeepNetwork(latent_dim=16, style_dim=128, num_domains=8, hidden_dim=512)
        self.style_dim=128
    
    def forward(self,x):

        z_trg = x['z_trg']
        y_org = x['y_trg']
        bs = y_org.shape[0]
        s_trg = self.mappingnet(z_trg, y_org).reshape(bs, -1, 128)
        return s_trg[:,0,:], s_trg[:,1:,:]

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from Utils.JDC.model import JDCNet
from modules.util import mydownres2Dblock
from modules.transformer import TransformerST, PositionalEncoding

class EmotionalDeformationTransformer(nn.Module): # EDN
    def __init__(self, embedding_dim, num_kp, num_w):
        super(EmotionalDeformationTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_kp = num_kp
        self.num_w = num_w

        self.pos_enc = PositionalEncoding(128,20)
        self.transformer = TransformerST()
        self.emokp = nn.Linear(128, 45)
        self.decode_dim = 32
        norm = 'batch'
        self.decodefeature_extract = nn.Sequential(mydownres2Dblock(self.decode_dim,32, normalize = norm),
                                        mydownres2Dblock(32,48, normalize = norm),
                                        mydownres2Dblock(48,64, normalize = norm),
                                        mydownres2Dblock(64,96, normalize = norm),
                                        mydownres2Dblock(96,128, normalize = norm),
                                        nn.AvgPool2d(2))
        
    def init_sidetuning_weight(self, state_dict):
        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                if 'decodefeature_extract.0.conv_block_0.layers' in name:
                    # print(len(param.shape), param.shape)
                    if len(param.shape)==1 and param.shape[0]==64:
                        own_state[name].copy_(param[32:])
                    elif len(param.shape)==4:
                        own_state[name].copy_(param[:,32:,:,])
                    else:
                        own_state[name].copy_(param)
                else:
                    own_state[name].copy_(param)

        total_num = sum(p.numel() for p in self.parameters())
        print(total_num)  # 52368368
        
    def forward(self, input_st, emoprompt, deepprompt):
        posi_em = self.pos_enc(7+1)
        bbs=input_st['bbs']
        bs=input_st['bs']
        prompt_feature = deepprompt.unsqueeze(1).tile(1, bs, 1, 1).reshape(bbs*bs, 6, 128)
        # print(prompt_feature.shape)
        emoprompt = emoprompt.unsqueeze(1).tile(1, bs, 1).reshape(bbs*bs, 1, 128)
        face_feature = input_st['face_feature_map']
        # print(input_st['memory'][11:,:,:].shape)
        face_feature = self.decodefeature_extract(face_feature).reshape(bbs, 1, -1).repeat(1, bs,1).reshape(bbs*bs, 1, -1)
        
        prompt_feature = torch.cat([face_feature, emoprompt, prompt_feature], dim=1)
        s_trg_feature = self.transformer(prompt_feature, posi_em)[1:] # -1 means the output of s_trg for emotion prediction
        s_trg_feature = torch.mean(s_trg_feature, dim=0)
        out= self.emokp(s_trg_feature)

        return out


