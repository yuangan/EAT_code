import torch.nn as nn
import torch
from modules.util import mydownres2Dblock
import numpy as np
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
import torch.nn.functional as F
import copy


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, winsize):
        return self.pos_table[:, :winsize].clone().detach()

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

### light weight transformer encoder
class TransformerST(nn.Module):

    def __init__(self, d_model=128, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        # flatten NxCxHxW to HWxNxC

        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)

        memory = self.encoder(src, pos=pos_embed)

        return memory

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask = None, src_key_padding_mask = None, pos = None):
        output = src+pos

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDeep(nn.Module):

    def __init__(self, d_model=128, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoderDeep(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoderDeep(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed, deepprompt):
        # flatten NxCxHxW to HWxNxC

        # print('src before permute: ', src.shape) # 5, 12, 128
        src = src.permute(1, 0, 2)
        # print('src after permute: ', src.shape) # 12, 5, 128
        pos_embed = pos_embed.permute(1, 0, 2)
        query_embed = query_embed.permute(1, 0, 2)

        tgt = torch.zeros_like(query_embed) # actually is tgt + query_embed
        memory = self.encoder(src, deepprompt, pos=pos_embed)

        hs = self.decoder(tgt, deepprompt, memory,
                          pos=pos_embed, query_pos=query_embed)
        return hs, memory

class TransformerEncoderDeep(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, deepprompt, mask = None, src_key_padding_mask = None, pos = None):
        # print('input: ', src.shape) # 12 5 128
        # print('deepprompt:', deepprompt.shape) # 1 6 128
        ### TODO: add deep prompt in encoder
        bs = src.shape[1]
        bbs = deepprompt.shape[0]
        idx=0
        emoprompt = deepprompt[:,idx,:]
        emoprompt = emoprompt.unsqueeze(1).tile(1, bs, 1).reshape(bbs*bs, 128).unsqueeze(0)
        # print(emoprompt.shape) # 1 5 128
        src = torch.cat([src, emoprompt], dim=0)
        # print(src.shape) # 13 5 128
        output = src+pos

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            ### deep prompt
            if idx+1 < len(self.layers):
                idx = idx + 1
                # print(idx)
                emoprompt = deepprompt[:,idx,:]
                emoprompt = emoprompt.unsqueeze(1).tile(1, bs, 1).reshape(bbs*bs, 128).unsqueeze(0)
                # print(output.shape) # 13 5 128
                output = torch.cat([output[:-1], emoprompt], dim=0)
                # print(output.shape) # 13 5 128

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderDeep(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, deepprompt, memory, tgt_mask = None,  memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        # print('input: ', query_pos.shape) 12 5 128
        ### TODO: add deep prompt in encoder
        bs = query_pos.shape[1]
        bbs = deepprompt.shape[0]
        idx=0
        emoprompt = deepprompt[:,idx,:]
        emoprompt = emoprompt.unsqueeze(1).tile(1, bs, 1).reshape(bbs*bs, 128).unsqueeze(0)
        query_pos = torch.cat([query_pos, emoprompt], dim=0)
        # print(query_pos.shape) # 13 5 128
        # print(torch.sum(tgt)) # 0
        output = pos+query_pos

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            
            ### deep prompt
            if idx+1 < len(self.layers):
                idx = idx + 1
                # print(idx)
                emoprompt = deepprompt[:,idx,:]
                emoprompt = emoprompt.unsqueeze(1).tile(1, bs, 1).reshape(bbs*bs, 128).unsqueeze(0)
                # print(output.shape) # 13 5 128
                output = torch.cat([output[:-1], emoprompt], dim=0)
                # print(output.shape) # 13 5 128

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask = None,
                     src_key_padding_mask = None,
                     pos = None):
        # q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(src, src, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask = None,
                    pos = None):
        src2 = self.norm1(src)
        # q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(src2, src2, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask = None,
                src_key_padding_mask = None,
                pos = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(tgt, tgt, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt,
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=tgt2,
                                   key=memory,
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

from Utils.JDC.model import JDCNet
from modules.audioencoder import AudioEncoder, MappingNetwork, StyleEncoder, AdaIN, EAModule

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, axis=1)
    # degree = F.one_hot(degree.to(torch.int64), num_classes=66)
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)

    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return yaw, pitch, roll, yaw_mat.view(yaw_mat.shape[0], 9), pitch_mat.view(pitch_mat.shape[0], 9), roll_mat.view(roll_mat.shape[0], 9), rot_mat.view(rot_mat.shape[0], 9)

class Audio2kpTransformerBBoxQDeepPrompt(nn.Module):
    def __init__(self, embedding_dim, num_kp, num_w, face_ea=False):
        super(Audio2kpTransformerBBoxQDeepPrompt, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_kp = num_kp
        self.num_w = num_w


        self.embedding = nn.Embedding(41, embedding_dim)

        self.face_shrink = nn.Linear(240, 32)
        self.hp_extractor = nn.Linear(45, 128)

        self.pos_enc = PositionalEncoding(128,20)
        input_dim = 1

        self.decode_dim = 64
        self.audio_embedding = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(29, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(192, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, 128),
        )

        self.audio_embedding2 = nn.Sequential(nn.Conv2d(1, 8, (3, 17), stride=(1, 1), padding=(1, 0)),
                                            #  nn.GroupNorm(4, 8, affine=True),
                                             BatchNorm2d(8),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(8, 32, (13, 13), stride=(1, 1), padding=(6, 6)))

        self.audioencoder = AudioEncoder(dim_in=64, style_dim=128, max_conv_dim=512, w_hpf=0, F0_channel=256)
        self.face_ea = face_ea
        if self.face_ea:
            self.fea = EAModule(style_dim=128, num_features=32)
        norm = 'batch'

        self.decodefeature_extract = nn.Sequential(mydownres2Dblock(self.decode_dim,32, normalize = norm),
                                             mydownres2Dblock(32,48, normalize = norm),
                                             mydownres2Dblock(48,64, normalize = norm),
                                             mydownres2Dblock(64,96, normalize = norm),
                                             mydownres2Dblock(96,128, normalize = norm),
                                             nn.AvgPool2d(2))

        self.feature_extract = nn.Sequential(mydownres2Dblock(input_dim,32),
                                             mydownres2Dblock(32,64),
                                             mydownres2Dblock(64,128),
                                             mydownres2Dblock(128,128),
                                             mydownres2Dblock(128,128),
                                             nn.AvgPool2d(2))
        self.transformer = TransformerDeep()
        self.kp = nn.Linear(128, 32)

        F0_path = './Utils/JDC/bst.t7'
        F0_model = JDCNet(num_class=1, seq_len=32)
        params = torch.load(F0_path, map_location='cpu')['net']
        F0_model.load_state_dict(params)
        self.f0_model = F0_model

    def rotation_and_translation(self, headpose, bbs, bs):
        yaw = headpose_pred_to_degree(headpose['yaw'].reshape(bbs*bs, -1))
        pitch = headpose_pred_to_degree(headpose['pitch'].reshape(bbs*bs, -1))
        roll = headpose_pred_to_degree(headpose['roll'].reshape(bbs*bs, -1))
        yaw_2, pitch_2, roll_2, yaw_v, pitch_v, roll_v, rot_v = get_rotation_matrix(yaw, pitch, roll)
        t = headpose['t'].reshape(bbs*bs, -1)
        hp = torch.cat([yaw.unsqueeze(1), pitch.unsqueeze(1), roll.unsqueeze(1), yaw_2, pitch_2, roll_2, yaw_v, pitch_v, roll_v, rot_v, t], dim=1)
        return hp

    def forward(self, x, initial_kp = None, return_strg=False, emoprompt=None, deepprompt=None, hp=None, side=False):
        bbs, bs, seqlen, _, _ = x['deep'].shape
        # ph = x["pho"].reshape(bbs*bs*seqlen, 1)
        if hp is None:
            hp = self.rotation_and_translation(x['he_driving'], bbs, bs)
        hp = self.hp_extractor(hp)

        pose_feature = x["pose"].reshape(bbs*bs*seqlen,1,64,64)

        audio = x['deep'].reshape(bbs*bs*seqlen, 16, 29).permute(0, 2, 1)
        deep_feature = self.audio_embedding(audio).squeeze(-1)# ([264, 32, 16, 16])

        input_feature = pose_feature
        input_feature = self.feature_extract(input_feature).reshape(bbs*bs*seqlen, 128)
        input_feature = torch.cat([input_feature, deep_feature], dim=1)
        input_feature = self.encoder_fc1(input_feature).reshape(bbs*bs, seqlen, 128)
        input_feature = torch.cat([input_feature, hp.unsqueeze(1)], dim=1)

        ### decode audio feature
        ### use iteration to avoid batchnorm2d in different audio sequence 
        decoder_features = []
        for i in range(bbs):
            F0 = self.f0_model.get_feature_GAN(x['mel'][i].reshape(bs, 1, 80, seqlen))
            if emoprompt is None:
                audio_feature = (self.audioencoder(x['mel'][i].reshape(bs, 1, 80, seqlen), s=None, masks=None, F0=F0))
            else:
                audio_feature = (self.audioencoder(x['mel'][i].reshape(bs, 1, 80, seqlen), s=emoprompt[i].unsqueeze(0), masks=None, F0=F0))
            audio2 = torch.permute(audio_feature, (0, 3, 1, 2)).reshape(bs*seqlen, 1, 64, 80)
            decoder_feature = self.audio_embedding2(audio2)

            face_map = initial_kp["prediction_map"][i].reshape(15*16, 64*64).permute(1, 0).reshape(64*64, 15*16)
            face_feature_map = self.face_shrink(face_map).permute(1, 0).reshape(1, 32, 64, 64)
            if self.face_ea:
                face_feature_map = self.fea(face_feature_map, emoprompt)
            decoder_feature = self.decodefeature_extract(torch.cat(
                (decoder_feature,
                face_feature_map.repeat(bs, seqlen, 1, 1, 1).reshape(bs * seqlen, 32, 64, 64)),
                dim=1)).reshape(bs, seqlen, 128)
            decoder_features.append(decoder_feature)
        decoder_feature = torch.cat(decoder_features, dim=0)
        
        decoder_feature = torch.cat([decoder_feature, hp.unsqueeze(1)], dim=1)

        # a2kp transformer
        # position embedding
        if emoprompt is None:
            posi_em = self.pos_enc(self.num_w*2+1+1) # 11 + headpose token
        else:
            posi_em = self.pos_enc(self.num_w*2+1+1+1) # 11 + headpose token + deep emotion prompt
        out = {}
        output_feature, memory = self.transformer(input_feature, decoder_feature, posi_em, deepprompt)
        output_feature = output_feature[-1, self.num_w] # returned intermediate output [6, 13, bbs*bs, 128]
        out["emo"] = self.kp(output_feature)
        if side:
            input_st = {}
            input_st['hp'] = hp
            input_st['face_feature_map'] = face_feature_map
            input_st['bs'] = bs
            input_st['bbs'] = bbs
            return out, input_st
        else:
            return out


