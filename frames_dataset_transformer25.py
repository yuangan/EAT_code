import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from os.path import dirname, join, basename, isfile
import json

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
import torch
import torchaudio
import random 
import glob
import soundfile as sf
import math
import cv2

from scipy.io import wavfile
import python_speech_features
# import pyworld
from scipy.interpolate import interp1d

### wav2lip read wav
from scipy import signal
import librosa
syncnet_mel_step_size = 16

###### CHANGE THE MEAD PATHS HERE ######
vox_path = '/data/gy/vox/'
mead_path = '/data5/gy/mead/'
poseimg_path = '/data2/gy/'


MEL_PARAMS_25 = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 640,
    "hop_length": 640
}

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array

import torch.nn.functional as F
def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, axis=1)
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
    return rot_mat

import gzip

### read in poseimg with gzip to save disk memory and train quicker
### wo readin image to train a2kp transformer
class FramesWavsDatasetMEL25VoxBoxQG2(Dataset):
    """
    Dataset of videos and wavs, each video can be represented as:
      - an image of concatenated frames
      - wavs
      - folder with all frames
    """
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, syncnet_T=24):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
        self.emo_label_list = ['Angry',  'Contempt',  'Disgust',  'Fear',  'Happy',  'Neutral',  'Sad',  'Surprised']

        self.length = 0
        if is_train:
            self.videos = train_videos
            self.length+=len(self.videos)
            # voxs dataset
            self.root_vox = f'{vox_path}/voxs_images/'
            # self.videos_vox = list({os.path.basename(video) for video in os.listdir(self.root_vox)})
            self.videos_vox_emo = {}
            self.videos_vox_len = {}
            self.indexes_vox = {}
            self.videos_vox_woemo = []
            for emo in self.emo_label_list:
                idlists = []
                with open(f'{vox_path}/voxselect/{emo}.txt', 'r') as femolist:
                    idtxt = femolist.readlines()
                    for i in idtxt:
                        idlists.append(i.split(' ')[0])
                idlists = idlists
                self.videos_vox_emo[emo] = idlists
                self.videos_vox_woemo.extend(idlists)
                self.videos_vox_len[emo] = len(idlists)
                self.indexes_vox[emo] = 0
                self.length += len(idlists)
            print(self.length)
            self.length = self.length
        else:
            self.videos = test_videos
            self.length+=len(self.videos)

        self.is_train = is_train
        self.latent_dim = 16
        self.num_w = 5

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
        self.emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
        self.emo_label_full = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS_25)
        self.mean, self.std = -4, 4
        self.syncnet_T = syncnet_T
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(idx)
        prob=0.1
        if self.is_train:
            if  random.random() > prob:
            # if  idx >= len(self.videos): #384
                # return self.getitem_vox_emo(idx)
                return self.getitem_vox_woemo(idx)
                # return self.getitem_neu(idx%len(self.videos))
            else:
                return self.getitem_neu(idx)
        else:
            return self.getitem_neu(idx)
        # return self.getitem_neu(idx)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame, he_driving, poseimg):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        he_d = {'yaw': [], 'pitch': [], 'roll': [], 't': [],'exp': []}

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{:04}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        for k in he_driving.keys():
            he_driving[k] = torch.from_numpy(he_driving[k])
            he_d[k] = he_driving[k][start_id - 1:start_id + self.syncnet_T - 1]

        # draw headpose
        poseboxs = []
        for i in range((start_id - 1 - self.num_w), (start_id + self.syncnet_T + self.num_w )):
            if i < 0:
                poseboxs.append(poseimg[0])
            elif i >= poseimg.shape[0]:
                poseboxs.append(poseimg[-1])
            else:
                poseboxs.append(poseimg[i])
        poseboxs = np.array(poseboxs)
        # print('poseboxs', poseboxs.shape) # self.syncnet_T+11
        return window_fnames, he_d, poseboxs

    def crop_audio_window(self, mel, poses, deeps, start_frame, num_frames):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame) - 1
        pad = torch.zeros((mel.shape[0]))
        # print(pad.shape)
        audio_frames = []
        pose_frames = []
        deep_frames = []

        zero_pad = np.zeros([self.num_w , deeps.shape[1], deeps.shape[2]])
        deeps = np.concatenate((zero_pad, deeps[:num_frames], zero_pad), axis=0)

        for rid in range(start_frame_num, start_frame_num + self.syncnet_T):
            audio = []
            for i in range(rid - self.num_w, rid + self.num_w + 1):
                if i < 0:
                    audio.append(pad)
                elif i >= num_frames:
                    audio.append(pad)
                else:
                    audio.append(mel[:, i])
            deep_frames.append(deeps[rid: rid+2*self.num_w+1])
            # if not len(deeps[rid: rid+2*self.num_w+1])==11:
                # print(start_frame_num, deeps.shape, num_frames, rid, self.syncnet_T)
                # assert(0)
            audio_frames.append(torch.stack(audio, dim=1))
            # print((rid - start_frame_num), (rid + 2*self.num_w + 1 - start_frame_num))
            pose_frames.append(poses[(rid - start_frame_num): (rid + 2*self.num_w + 1 - start_frame_num)])
        audio_f = torch.stack(audio_frames, dim=0)
        poses_f = torch.from_numpy(np.array(pose_frames, dtype=np.float32))
        # print(poses_f.shape)
        # print(audio_f.shape)
        # assert(0)
        deep_frames = torch.from_numpy(np.array(deep_frames)).to(torch.float)

        return audio_f, poses_f, deep_frames

    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def getitem_neu(self, idx):
        while 1:
            idx = idx%len(self.videos)
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

            video_name = os.path.basename(path)
            vsp = video_name.split('_')

            out = {}
            
            deep_path = f'{mead_path}/deepfeature32/{video_name}.npy'
            deeps = np.load(deep_path)

            wave_path = f'{mead_path}/wav_16000/{video_name}.wav'
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std

            lable_index = self.emo_label.index(vsp[1])

            # print(out['drivinglmk'].shape)


            # out['y_trg'] = self.emo_label.index(vsp[1])
            # z_trg = torch.randn(self.latent_dim)
            # out['z_trg'] = z_trg

            # select gt frames
            frames = os.listdir(path)
            num_frames = len(frames)
            num_frames = min(num_frames, len(deeps), mel_tensor.shape[1])

            if num_frames - self.syncnet_T + 1 <= 0:
                # print(num_frames)
                idx += 1
                idx = idx%len(self.videos)
                continue
            frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0]+1
            choose = join(path, '{:04}.jpg'.format(frame_idx))
            
            # driving latent with pretrained
            driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
            he_driving = driving_latent[1]
            
            ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
            fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{video_name}.npy.gz', "r")
            poseimg = np.load(fposeimg)


            try:
                window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
            except:
                print(choose, path)
                idx += 1
                idx = idx%len(self.videos)
                continue
            out['he_driving'] = he_d
            
            # neutral frames
            video_name_neu = vsp[0]+'_neu_1_'+'*'
            path_neus = path.replace(video_name, video_name_neu)
            path_neu = random.choice(glob.glob(path_neus))
            source_latent = np.load(path_neu.replace('images', 'latent')+'.npy', allow_pickle=True)
            num_frames_source = source_latent[1]['yaw'].shape[0]
            source_index=np.random.choice(num_frames_source, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))
            
            # neutral source latent with pretrained
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = torch.from_numpy(source_latent[1][k][source_index-1])
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))




            mel, poses_f, deep_frames = self.crop_audio_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = video_name
            out['deep'] = deep_frames

            return out
    
    def getitem_vox_woemo(self, index):
        while 1:
            if self.is_train and self.id_sampling:
                assert(0)
            else:
                
                r = np.random.choice(len(self.videos_vox_woemo), replace=True, size=1)[0]

                # iter done, clear vox indexes
                name = self.videos_vox_woemo[r]
                path = os.path.join(self.root_vox, name)
                wave_path = f'{vox_path}/voxs_wavs/{name}.wav'
                
                if not os.path.exists(path) or not os.path.exists(wave_path):
                    print(path, r, wave_path)
                    continue

            # print(path)
            # self.count += 1
            out = {}
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
            # print(out['drivinglmk'].shape)
            try:
                deep_path = f'{vox_path}/deepfeature32/{name}.npy'
                deeps = np.load(deep_path)
            except:
                print(f'{vox_path}/deepfeature32/{name}.npy')
                continue
            
            # driving latent with pretrained
            driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
            he_driving = driving_latent[1]

            ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
            fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{name}.npy.gz', "r")
            poseimg = np.load(fposeimg)

            # select gt frames
            # frames = glob.glob(path+'/*.jpg')
            num_frames = he_driving['yaw'].shape[0]
            num_frames = min(num_frames, len(deeps), mel_tensor.shape[1])

            if num_frames- self.syncnet_T+1 <= 0:
                # print(name, num_frames)
                continue
            frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0]+1
            choose = join(path, '{:04}.jpg'.format(frame_idx))

            # try:
            window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
            # window_fnames, he_d = self.get_window(choose, he_driving)
            # except:
            #     print(choose, path)
            #     continue
            out['he_driving'] = he_d

            # neutral frames
            path_neu = path
            source_index=np.random.choice(num_frames, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))

            # neutral source latent with pretrained
            source_latent = driving_latent
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = source_latent[1][k][source_index-1]
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))
            

            mel, poses_f, deep_frames = self.crop_audio_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = name
            out['deep'] = deep_frames

            return out



### read in poseimg with gzip to save disk memory and train quicker
class FramesWavsDatasetMEL25VoxBoxQG2ImgAll(Dataset):
    """
    Dataset of videos and wavs, each video can be represented as:
      - an image of concatenated frames
      - wavs
      - folder with all frames
    """
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, syncnet_T=24):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
        self.emo_label_list = ['Angry',  'Contempt',  'Disgust',  'Fear',  'Happy',  'Neutral',  'Sad',  'Surprised']

        self.length = 0
        if is_train:
            self.videos = train_videos
            self.length+=len(self.videos)
            # voxs dataset
            self.root_vox = f'{vox_path}/voxs_images/'
            # self.videos_vox = list({os.path.basename(video) for video in os.listdir(self.root_vox)})
            self.videos_vox_emo = {}
            self.videos_vox_len = {}
            self.indexes_vox = {}
            self.videos_vox_woemo = []
            for emo in self.emo_label_list:
                idlists = []
                with open(f'{vox_path}/voxselect/{emo}.txt', 'r') as femolist:
                    idtxt = femolist.readlines()
                    for i in idtxt:
                        idlists.append(i.split(' ')[0])
                idlists = idlists
                self.videos_vox_emo[emo] = idlists
                self.videos_vox_woemo.extend(idlists)
                self.videos_vox_len[emo] = len(idlists)
                self.indexes_vox[emo] = 0
                self.length += len(idlists)//2
            self.length = self.length
            print(self.length)
        else:
            self.videos = test_videos
            self.length+=len(self.videos)

        self.is_train = is_train
        self.latent_dim = 16
        self.num_w = 5

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
        self.emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
        self.emo_label_full = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS_25)
        self.mean, self.std = -4, 4
        self.syncnet_T = syncnet_T
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        prob=0.1
        if self.is_train:
            if  random.random() > prob:
            # if  idx >= len(self.videos): #384
                # return self.getitem_vox_emo(idx)
                return self.getitem_vox_woemo(idx)
                # return self.getitem_neu(idx%len(self.videos))
            else:
                return self.getitem_neu(idx)
        else:
            return self.getitem_neu(idx)
        # return self.getitem_neu(idx)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame, he_driving, poseimg):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        he_d = {'yaw': [], 'pitch': [], 'roll': [], 't': [],'exp': []}

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{:04}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        for k in he_driving.keys():
            he_driving[k] = torch.from_numpy(he_driving[k])
            he_d[k] = he_driving[k][start_id - 1:start_id + self.syncnet_T - 1]

        # draw headpose
        poseboxs = []
        for i in range((start_id - 1 - self.num_w), (start_id + self.syncnet_T + self.num_w )):
            if i < 0:
                poseboxs.append(poseimg[0])
            elif i >= poseimg.shape[0]:
                poseboxs.append(poseimg[-1])
            else:
                poseboxs.append(poseimg[i])
        poseboxs = np.array(poseboxs)
        # print('poseboxs', poseboxs.shape) # self.syncnet_T+11
        return window_fnames, he_d, poseboxs

    def crop_audio_phoneme_window(self, mel, poses, deeps, start_frame, num_frames):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame) - 1
        pad = torch.zeros((mel.shape[0]))
        # print(pad.shape)
        audio_frames = []
        pose_frames = []
        deep_frames = []

        zero_pad = np.zeros([self.num_w , deeps.shape[1], deeps.shape[2]])
        deeps = np.concatenate((zero_pad, deeps[:num_frames], zero_pad), axis=0)

        for rid in range(start_frame_num, start_frame_num + self.syncnet_T):
            audio = []
            for i in range(rid - self.num_w, rid + self.num_w + 1):
                if i < 0:
                    audio.append(pad)
                elif i >= num_frames:
                    audio.append(pad)
                else:
                    audio.append(mel[:, i])
            deep_frames.append(deeps[rid: rid+2*self.num_w+1])
            # if not len(deeps[rid: rid+2*self.num_w+1])==11:
                # print(start_frame_num, deeps.shape, num_frames, rid, self.syncnet_T)
                # assert(0)
            audio_frames.append(torch.stack(audio, dim=1))
            # print((rid - start_frame_num), (rid + 2*self.num_w + 1 - start_frame_num))
            pose_frames.append(poses[(rid - start_frame_num): (rid + 2*self.num_w + 1 - start_frame_num)])
        audio_f = torch.stack(audio_frames, dim=0)
        poses_f = torch.from_numpy(np.array(pose_frames, dtype=np.float32))
        # print(poses_f.shape)
        # print(audio_f.shape)
        # assert(0)
        deep_frames = torch.from_numpy(np.array(deep_frames)).to(torch.float)

        return audio_f, poses_f, deep_frames

    def get_sync_mel(self, wav, start_frame):
        def preemphasis(wav, k, preemphasize=True):
            if preemphasize:
                return signal.lfilter([1, -k], [1], wav)
            return wav

        def _stft(y):
            return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)

        def _amp_to_db(x):
            min_level = np.exp(-100 / 20 * np.log(10))
            return 20 * np.log10(np.maximum(min_level, x))
        
        def _linear_to_mel(spectogram):
            _mel_basis = _build_mel_basis()
            return np.dot(_mel_basis, spectogram)
        
        def _build_mel_basis():
            # assert 7600 <= 16000 // 2
            return librosa.filters.mel(16000, 800, n_mels=80,
                                    fmin=55, fmax=7600)

        def _normalize(S, max_abs_value=4., min_level_db=-100):
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                        -max_abs_value, max_abs_value)
                
        D = _stft(preemphasis(wav, 0.97))
        S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
        
        S = _normalize(S).T

        fps = 25
        start_idx = int(80. * (start_frame / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return S[start_idx : end_idx, :]


    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def getitem_neu(self, idx):
        while 1:
            idx = np.random.choice(len(self.videos), replace=True, size=1)[0]
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

            video_name = os.path.basename(path)
            vsp = video_name.split('_')

            out = {}
            
            deep_path = f'{mead_path}/deepfeature32/{video_name}.npy'
            deeps = np.load(deep_path)

            wave_path = f'{mead_path}/wav_16000/{video_name}.wav'
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std

            lable_index = self.emo_label.index(vsp[1])
            emoname = self.emo_label_full[lable_index]

            ### bboxs files extracted
            bboxs = np.load(f'{mead_path}/bboxs/{video_name}.npy', allow_pickle=True)
            
            # print(out['drivinglmk'].shape)


            # out['y_trg'] = self.emo_label.index(vsp[1])
            # z_trg = torch.randn(self.latent_dim)
            # out['z_trg'] = z_trg

            # select gt frames
            frames = os.listdir(path)
            num_frames = len(frames)
            num_frames = min(num_frames, len(deeps), mel_tensor.shape[1], len(bboxs))

            if num_frames - self.syncnet_T + 1 <= 0:
                # print(num_frames)
                idx += 1
                continue
            frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0] + 1
            choose = join(path, '{:04}.jpg'.format(frame_idx))
            
            # driving latent with pretrained
            driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
            he_driving = driving_latent[1]
            
            ### get syncmel refer to: wav2lip
            ## The syncnet_T length should be 5 or don't use the sync_mel_tensor
            ## fps: 25 wav: 16k Hz 
            ## sync_mel_tensor shape: [16, 80]
            ## cost: 0.05s
            sync_mel_tensor = self.get_sync_mel(wave_tensor, frame_idx-1).T
            out['sync_mel'] = np.expand_dims(sync_mel_tensor, axis=0).astype(np.float32)

            ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
            fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{video_name}.npy.gz', "r")
            poseimg = np.load(fposeimg)



            try:
                window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
            except:
                print(choose, path)
                idx += 1
                idx = idx%len(self.videos)
                continue
            out['he_driving'] = he_d
            
            ### read img 
            ## cost 0.2s for 5 frames
            window = []
            boxs = []
            all_read = True
            count = 0
            for fname in window_fnames:
                img = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                box = bboxs[frame_idx-1+count]
                if box is None:
                    all_read = False
                    break
                if img is None:
                    all_read = False
                    break
                window.append(img)
                boxs.append(box)
                count += 1
            if not all_read: 
                idx += 1
                idx = idx%len(self.videos)
                continue
            out['driving'] = np.stack(window, axis=0)
            out['bboxs'] = np.stack(boxs, axis=0)
            # print(out['window'].shape)


            # neutral frames
            video_name_neu = vsp[0]+'_neu_1_'+'*'
            path_neus = path.replace(video_name, video_name_neu)
            path_neu = random.choice(glob.glob(path_neus))
            source_latent = np.load(path_neu.replace('images', 'latent')+'.npy', allow_pickle=True)
            num_frames_source = source_latent[1]['yaw'].shape[0]
            source_index=np.random.choice(num_frames_source, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))
            
            # neutral source latent with pretrained
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = torch.from_numpy(source_latent[1][k][source_index-1])
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))




            mel, poses_f, deep_frames = self.crop_audio_phoneme_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = video_name
            out['deep'] = deep_frames

            return out
    
    def getitem_vox_woemo(self, index):
        while 1:
            if self.is_train and self.id_sampling:
                assert(0)
            else:
                
                r = np.random.choice(len(self.videos_vox_woemo), replace=True, size=1)[0]

                # iter done, clear vox indexes
                name = self.videos_vox_woemo[r]
                path = os.path.join(self.root_vox, name)
                wave_path = f'{vox_path}/voxs_wavs/{name}.wav'
                
                if not os.path.exists(path) or not os.path.exists(wave_path):
                    print(path, r, wave_path)
                    continue

            out = {}
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
            # print(out['drivinglmk'].shape)
            try:
                deep_path = f'{vox_path}/deepfeature32/{name}.npy'
                deeps = np.load(deep_path)

            
                # driving latent with pretrained
                driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
                he_driving = driving_latent[1]


                ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
                fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{name}.npy.gz', "r")
                poseimg = np.load(fposeimg)

                ### bboxs files extracted
                bboxs = np.load(f'{vox_path}/bboxs/{name}.npy', allow_pickle=True)
            
                # select gt frames
                # frames = glob.glob(path+'/*.jpg')
                num_frames = he_driving['yaw'].shape[0]
                num_frames = min(num_frames, len(deeps), mel_tensor.shape[1], len(bboxs))

                if num_frames- self.syncnet_T+1 <= 0:
                    # print(name, num_frames)
                    continue
                
                frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0]+1
                choose = join(path, '{:04}.jpg'.format(frame_idx))

                ### get syncmel refer to: wav2lip
                ## The syncnet_T length should be 5 or don't use the sync_mel_tensor
                ## fps: 25 wav: 16k Hz 
                ## sync_mel_tensor shape: [16, 80]
                ## cost: 0.05s
                sync_mel_tensor = self.get_sync_mel(wave_tensor, frame_idx-1).T
                out['sync_mel'] = np.expand_dims(sync_mel_tensor, axis=0).astype(np.float32)

                # try:
                window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
                # window_fnames, he_d = self.get_window(choose, he_driving)
                # except:
                #     print(choose, path)
                #     continue
                out['he_driving'] = he_d

                ### read img 
                ## cost 0.2s for 5 frames
                window = []
                boxs = []
                all_read = True
                count = 0
                for fname in window_fnames:
                    img = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                    box = bboxs[frame_idx-1+count]
                    if box is None:
                        all_read = False
                        break
                    if img is None:
                        all_read = False
                        break
                    window.append(img)
                    boxs.append(box)
                    count += 1
                if not all_read:
                    continue
                out['driving'] = np.stack(window, axis=0)
                out['bboxs'] = np.stack(boxs, axis=0)
            # print(out['window'].shape)
            except:
                print(deep_path)
                continue
            # neutral frames
            path_neu = path
            source_index=np.random.choice(num_frames, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))

            # neutral source latent with pretrained
            source_latent = driving_latent
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = source_latent[1][k][source_index-1]
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))
            

            mel, poses_f, deep_frames = self.crop_audio_phoneme_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = name
            out['deep'] = deep_frames

            return out

### read in poseimg with gzip to save disk memory and train quicker
### add getitem_vox_emo
class FramesWavsDatasetMEL25VoxBoxQG2ImgPrompt(Dataset):
    """
    Dataset of videos and wavs, each video can be represented as:
      - an image of concatenated frames
      - wavs
      - folder with all frames
    """
    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None, syncnet_T=24, use_vox=False):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
        self.emo_label_list = ['Angry',  'Contempt',  'Disgust',  'Fear',  'Happy',  'Neutral',  'Sad',  'Surprised']

        self.length = 0
        if is_train:
            self.videos = train_videos
            self.length+=len(self.videos)
            # voxs dataset
            self.root_vox = f'{vox_path}/voxs_images/'
            # self.videos_vox = list({os.path.basename(video) for video in os.listdir(self.root_vox)})
            self.videos_vox_emo = {}
            self.videos_vox_len = {}
            # self.indexes_vox = {}
            # self.videos_vox_woemo = []
            for emo in self.emo_label_list:
                idlists = []
                with open(f'{vox_path}/voxselect/{emo}.txt', 'r') as femolist:
                    idtxt = femolist.readlines()[:1000]
                    for i in idtxt:
                        idlists.append(i.split(' ')[0])
                idlists = idlists
                self.videos_vox_emo[emo] = idlists
                # self.videos_vox_woemo.extend(idlists)
                # self.indexes_vox[emo] = 0
                self.videos_vox_len[emo] = len(idlists)
                if use_vox:
                    self.length += len(idlists) # use vox
            print(self.length)
            self.length = self.length
        else:
            self.videos = test_videos
            self.length+=len(self.videos)

        self.is_train = is_train
        self.latent_dim = 16
        self.num_w = 5

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
        self.emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
        self.emo_label_full = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS_25)
        self.mean, self.std = -4, 4
        self.syncnet_T = syncnet_T
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_train:
            if  idx >= len(self.videos): #384
                return self.getitem_vox_emo(idx)
                # return self.getitem_vox_woemo(idx)
                # return self.getitem_neu(idx%len(self.videos))
            else:
                return self.getitem_neu(idx)
        else:
            return self.getitem_neu(idx)
        # return self.getitem_neu(idx)

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame, he_driving, poseimg):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        he_d = {'yaw': [], 'pitch': [], 'roll': [], 't': [],'exp': []}

        window_fnames = []
        for frame_id in range(start_id, start_id + self.syncnet_T):
            frame = join(vidname, '{:04}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        for k in he_driving.keys():
            he_driving[k] = torch.from_numpy(he_driving[k])
            he_d[k] = he_driving[k][start_id - 1:start_id + self.syncnet_T - 1]

        # draw headpose
        poseboxs = []
        for i in range((start_id - 1 - self.num_w), (start_id + self.syncnet_T + self.num_w )):
            if i < 0:
                poseboxs.append(poseimg[0])
            elif i >= poseimg.shape[0]:
                poseboxs.append(poseimg[-1])
            else:
                poseboxs.append(poseimg[i])
        poseboxs = np.array(poseboxs)
        # print('poseboxs', poseboxs.shape) # self.syncnet_T+11
        return window_fnames, he_d, poseboxs

    def crop_audio_phoneme_window(self, mel, poses, deeps, start_frame, num_frames):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame) - 1
        pad = torch.zeros((mel.shape[0]))
        # print(pad.shape)
        audio_frames = []
        pose_frames = []
        deep_frames = []

        zero_pad = np.zeros([self.num_w , deeps.shape[1], deeps.shape[2]])
        deeps = np.concatenate((zero_pad, deeps[:num_frames], zero_pad), axis=0)

        for rid in range(start_frame_num, start_frame_num + self.syncnet_T):
            audio = []
            for i in range(rid - self.num_w, rid + self.num_w + 1):
                if i < 0:
                    audio.append(pad)
                elif i >= num_frames:
                    audio.append(pad)
                else:
                    audio.append(mel[:, i])
            deep_frames.append(deeps[rid: rid+2*self.num_w+1])
            # if not len(deeps[rid: rid+2*self.num_w+1])==11:
                # print(start_frame_num, deeps.shape, num_frames, rid, self.syncnet_T)
                # assert(0)
            audio_frames.append(torch.stack(audio, dim=1))
            # print((rid - start_frame_num), (rid + 2*self.num_w + 1 - start_frame_num))
            pose_frames.append(poses[(rid - start_frame_num): (rid + 2*self.num_w + 1 - start_frame_num)])
        audio_f = torch.stack(audio_frames, dim=0)
        poses_f = torch.from_numpy(np.array(pose_frames, dtype=np.float32))
        # print(poses_f.shape)
        # print(audio_f.shape)
        # assert(0)
        deep_frames = torch.from_numpy(np.array(deep_frames)).to(torch.float)

        return audio_f, poses_f, deep_frames

    def get_sync_mel(self, wav, start_frame):
        def preemphasis(wav, k, preemphasize=True):
            if preemphasize:
                return signal.lfilter([1, -k], [1], wav)
            return wav

        def _stft(y):
            return librosa.stft(y=y, n_fft=800, hop_length=200, win_length=800)

        def _amp_to_db(x):
            min_level = np.exp(-100 / 20 * np.log(10))
            return 20 * np.log10(np.maximum(min_level, x))
        
        def _linear_to_mel(spectogram):
            _mel_basis = _build_mel_basis()
            return np.dot(_mel_basis, spectogram)
        
        def _build_mel_basis():
            # assert 7600 <= 16000 // 2
            return librosa.filters.mel(16000, 800, n_mels=80,
                                    fmin=55, fmax=7600)

        def _normalize(S, max_abs_value=4., min_level_db=-100):
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                        -max_abs_value, max_abs_value)
                
        D = _stft(preemphasis(wav, 0.97))
        S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
        
        S = _normalize(S).T

        fps = 25
        start_idx = int(80. * (start_frame / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        return S[start_idx : end_idx, :]


    def _load_tensor(self, data):
        wave_path = data
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor

    def getitem_neu(self, idx):
        while 1:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

            video_name = os.path.basename(path)
            vsp = video_name.split('_')

            out = {}
            
            out['y_trg'] = self.emo_label.index(vsp[1])
            z_trg = torch.randn(self.latent_dim)
            out['z_trg'] = z_trg

            deep_path = f'{mead_path}/deepfeature32/{video_name}.npy'
            deeps = np.load(deep_path)

            wave_path = f'{mead_path}/wav_16000/{video_name}.wav'
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std

            lable_index = self.emo_label.index(vsp[1])
            emoname = self.emo_label_full[lable_index]

            # select gt frames
            frames = os.listdir(path)
            num_frames = len(frames)
            num_frames = min(num_frames, len(deeps), mel_tensor.shape[1])

            if num_frames - self.syncnet_T + 1 <= 0:
                # print(num_frames)
                idx += 1
                idx = idx%len(self.videos)
                continue
            frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0] + 1
            choose = join(path, '{:04}.jpg'.format(frame_idx))
            
            # driving latent with pretrained
            driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
            he_driving = driving_latent[1]
            
            ### get syncmel refer to: wav2lip
            ## The syncnet_T length should be 5 or don't use the sync_mel_tensor
            ## fps: 25 wav: 16k Hz 
            ## sync_mel_tensor shape: [16, 80]
            ## cost: 0.05s
            sync_mel_tensor = self.get_sync_mel(wave_tensor, frame_idx-1).T
            out['sync_mel'] = np.expand_dims(sync_mel_tensor, axis=0).astype(np.float32)

            ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
            fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{video_name}.npy.gz', "r")
            poseimg = np.load(fposeimg)

            ### bboxs files extracted
            bboxs = np.load(f'{mead_path}/bboxs/{video_name}.npy', allow_pickle=True)

            try:
                window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
            except:
                print(choose, path)
                idx += 1
                idx = idx%len(self.videos)
                continue
            out['he_driving'] = he_d
            
            ### read img 
            ## cost 0.2s for 5 frames
            window = []
            boxs = []
            all_read = True
            count = 0
            for fname in window_fnames:
                img = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                box = bboxs[frame_idx-1+count]
                if box is None:
                    all_read = False
                    break
                if img is None:
                    all_read = False
                    break
                window.append(img)
                boxs.append(box)
                count += 1
            if not all_read: 
                idx += 1
                idx = idx%len(self.videos)
                continue
            out['driving'] = np.stack(window, axis=0)
            out['bboxs'] = np.stack(boxs, axis=0)
            # print(out['window'].shape)


            # neutral frames
            video_name_neu = vsp[0]+'_neu_1_'+'*'
            path_neus = path.replace(video_name, video_name_neu)
            path_neu = random.choice(glob.glob(path_neus))
            source_latent = np.load(path_neu.replace('images', 'latent')+'.npy', allow_pickle=True)
            num_frames_source = source_latent[1]['yaw'].shape[0]
            source_index=np.random.choice(num_frames_source, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))
            
            # neutral source latent with pretrained
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = torch.from_numpy(source_latent[1][k][source_index-1])
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))


            mel, poses_f, deep_frames = self.crop_audio_phoneme_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = video_name
            out['deep'] = deep_frames

            return out

    def getitem_vox_emo(self, index):
        while 1:
            if self.is_train and self.id_sampling:
                assert(0)
            else:
                
                emo_r = np.random.choice(8, replace=True, size=1)[0]
                emo_name = self.emo_label_list[emo_r]
                r = np.random.choice(len(self.videos_vox_emo[emo_name]), replace=True, size=1)[0]

                # iter done, clear vox indexes
                name = self.videos_vox_emo[emo_name][r]
                path = os.path.join(self.root_vox, name)
                wave_path = f'{vox_path}/voxs_wavs/{name}.wav'
                
                if not os.path.exists(path) or not os.path.exists(wave_path):
                    print(path, r, wave_path)
                    continue

            out = {}

            out['y_trg'] = emo_r
            z_trg = torch.randn(self.latent_dim)
            out['z_trg'] = z_trg

            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
            # print(out['drivinglmk'].shape)
            try:
                deep_path = f'{vox_path}/deepfeature32/{name}.npy'
                deeps = np.load(deep_path)
            except:
                print(deep_path)
                continue
            
            # driving latent with pretrained
            driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
            he_driving = driving_latent[1]


            ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
            fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{name}.npy.gz', "r")
            poseimg = np.load(fposeimg)

            ### bboxs files extracted
            bboxs = np.load(f'{vox_path}/bboxs/{name}.npy', allow_pickle=True)

            # select gt frames
            # frames = glob.glob(path+'/*.jpg')
            num_frames = he_driving['yaw'].shape[0]
            num_frames = min(num_frames, len(deeps), mel_tensor.shape[1])

            if num_frames- self.syncnet_T+1 <= 0:
                # print(name, num_frames)
                continue
            
            frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0]+1
            choose = join(path, '{:04}.jpg'.format(frame_idx))

            ### get syncmel refer to: wav2lip
            ## The syncnet_T length should be 5 or don't use the sync_mel_tensor
            ## fps: 25 wav: 16k Hz 
            ## sync_mel_tensor shape: [16, 80]
            ## cost: 0.05s
            sync_mel_tensor = self.get_sync_mel(wave_tensor, frame_idx-1).T
            out['sync_mel'] = np.expand_dims(sync_mel_tensor, axis=0).astype(np.float32)
            # print(num_frames, len(deeps), mel_tensor.shape[1], out['sync_mel'].shape)
            # try:
            window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
            # window_fnames, he_d = self.get_window(choose, he_driving)
            # except:
            #     print(choose, path)
            #     continue
            out['he_driving'] = he_d

            ### read img 
            ## cost 0.2s for 5 frames
            window = []
            boxs = []
            all_read = True
            count = 0
            for fname in window_fnames:
                img = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                box = bboxs[frame_idx-1+count]
                if box is None:
                    print(fname)
                    all_read = False
                    break
                else:
                    wbox = box[2] - box[0]
                    hbox = box[3] - box[1]
                    if wbox<100 or hbox<100:
                        print(fname)
                        print(box)
                        all_read=False
                        break
                if img is None:
                    all_read = False
                    break
                window.append(img)
                boxs.append(box)
                count += 1
            if not all_read:
                continue
            out['driving'] = np.stack(window, axis=0)
            out['bboxs'] = np.stack(boxs, axis=0)
            # print(out['window'].shape)

            # neutral frames
            path_neu = path
            source_index=np.random.choice(num_frames, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))

            # neutral source latent with pretrained
            source_latent = driving_latent
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = source_latent[1][k][source_index-1]
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))
            

            mel, poses_f, deep_frames = self.crop_audio_phoneme_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = name
            out['deep'] = deep_frames

            return out

    def getitem_vox_woemo(self, index):
        while 1:
            if self.is_train and self.id_sampling:
                assert(0)
            else:
                
                r = np.random.choice(len(self.videos_vox_woemo), replace=True, size=1)[0]

                # iter done, clear vox indexes
                name = self.videos_vox_woemo[r]
                path = os.path.join(self.root_vox, name)
                wave_path = f'{vox_path}/voxs_wavs/{name}.wav'
                
                if not os.path.exists(path) or not os.path.exists(wave_path):
                    print(path, r, wave_path)
                    continue

            out = {}
            out['wave_path'] = wave_path
            wave_tensor = self._load_tensor(wave_path)
            if len(wave_tensor.shape) > 1:
                wave_tensor = wave_tensor[:, 0]
            mel_tensor = self.to_melspec(wave_tensor)
            mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
            # print(out['drivinglmk'].shape)
            try:
                deep_path = f'{vox_path}/deepfeature32/{name}.npy'
                deeps = np.load(deep_path)
            except:
                print(deep_path)
                continue
            
            # driving latent with pretrained
            driving_latent = np.load(path.replace('images', 'latent')+'.npy', allow_pickle=True)
            he_driving = driving_latent[1]


            ### poseimg after AntiAliasInterpolation2d: num_frames, 1, 64, 64
            fposeimg = gzip.GzipFile(f'{poseimg_path}/poseimg/{name}.npy.gz', "r")
            poseimg = np.load(fposeimg)

            ### bboxs files extracted
            bboxs = np.load(f'{vox_path}/bboxs/{name}.npy', allow_pickle=True)

            # select gt frames
            # frames = glob.glob(path+'/*.jpg')
            num_frames = he_driving['yaw'].shape[0]
            num_frames = min(num_frames, len(deeps), mel_tensor.shape[1])

            if num_frames- self.syncnet_T+1 <= 0:
                # print(name, num_frames)
                continue
            
            frame_idx = np.random.choice(num_frames- self.syncnet_T+1, replace=True, size=1)[0]+1
            choose = join(path, '{:04}.jpg'.format(frame_idx))

            ### get syncmel refer to: wav2lip
            ## The syncnet_T length should be 5 or don't use the sync_mel_tensor
            ## fps: 25 wav: 16k Hz 
            ## sync_mel_tensor shape: [16, 80]
            ## cost: 0.05s
            sync_mel_tensor = self.get_sync_mel(wave_tensor, frame_idx-1).T
            out['sync_mel'] = np.expand_dims(sync_mel_tensor, axis=0).astype(np.float32)
            # print(num_frames, len(deeps), mel_tensor.shape[1], out['sync_mel'].shape)
            # try:
            window_fnames, he_d, poses = self.get_window(choose, he_driving, poseimg)
            # window_fnames, he_d = self.get_window(choose, he_driving)
            # except:
            #     print(choose, path)
            #     continue
            out['he_driving'] = he_d

            ### read img 
            ## cost 0.2s for 5 frames
            window = []
            boxs = []
            all_read = True
            count = 0
            for fname in window_fnames:
                img = img_as_float32(io.imread(fname)).transpose((2, 0, 1))
                box = bboxs[frame_idx-1+count]
                if box is None:
                    print(fname)
                    all_read = False
                    break
                else:
                    wbox = box[2] - box[0]
                    hbox = box[3] - box[1]
                    if wbox<100 or hbox<100:
                        print(fname)
                        print(box)
                        all_read=False
                        break
                if img is None:
                    all_read = False
                    break
                window.append(img)
                boxs.append(box)
                count += 1
            if not all_read:
                continue
            out['driving'] = np.stack(window, axis=0)
            out['bboxs'] = np.stack(boxs, axis=0)
            # print(out['window'].shape)

            # neutral frames
            path_neu = path
            source_index=np.random.choice(num_frames, replace=True, size=1)[0]+1
            video_array_source = img_as_float32(io.imread(join(path_neu, '{:04}.jpg'.format(source_index))))

            # neutral source latent with pretrained
            source_latent = driving_latent
            he_source = {}
            for k in source_latent[1].keys():
                he_source[k] = source_latent[1][k][source_index-1]
            out['he_source'] = he_source

            out['source'] = video_array_source.transpose((2, 0, 1))
            

            mel, poses_f, deep_frames = self.crop_audio_phoneme_window(mel_tensor, poses, deeps, choose, num_frames)
            out['mel'] = mel.unsqueeze(1)
            out['pose'] = poses_f
            out['name'] = name
            out['deep'] = deep_frames

            return out

def draw_annotation_box( image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
    """Draw a 3D box as annotation of pose"""

    camera_matrix = np.array(
        [[233.333, 0, 128],
         [0, 233.333, 128],
         [0, 0, 1]], dtype="double")

    dist_coeefs = np.zeros((4, 1))

    point_3d = []
    rear_size = 75
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 100
    front_depth = 100
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)

def parse_phoneme_file(phoneme_path, use_index = True, mel_fps=30):
    with open(phoneme_path,'r') as f:
        result_text = json.load(f)
    frame_num = math.ceil(result_text[-1]['phones'][-1]['ed']/100*mel_fps)
    # print('frame_num of phonemes: ', frame_num)
    phoneset_list = []
    index = 0

    word_len = len(result_text)
    word_index = 0
    phone_index = 0
    cur_phone_list = result_text[0]["phones"]
    phone_len = len(cur_phone_list)
    cur_end = cur_phone_list[0]["ed"]

    phone_list = []

    phoneset_list.append(cur_phone_list[0]["ph"])
    i = 0
    while i < frame_num:
        if i * 100/mel_fps < cur_end:
            phone_list.append(cur_phone_list[phone_index]["ph"])
            i += 1
        else:
            phone_index += 1
            if phone_index >= phone_len:
                word_index += 1
                if word_index >= word_len:
                    phone_list.append(cur_phone_list[-1]["ph"])
                    i += 1
                else:
                    phone_index = 0
                    cur_phone_list = result_text[word_index]["phones"]
                    phone_len = len(cur_phone_list)
                    cur_end = cur_phone_list[phone_index]["ed"]
                    phoneset_list.append(cur_phone_list[phone_index]["ph"])
                    index += 1
            else:
                # print(word_index,phone_index)
                cur_end = cur_phone_list[phone_index]["ed"]
                phoneset_list.append(cur_phone_list[phone_index]["ph"])
                index += 1

    with open("phindex.json") as f:
        ph2index = json.load(f)
    if use_index:
        phone_list = [ph2index[p] for p in phone_list]
    # print('func:parse_phoneme_file: phone length: ', len(phone_list))
    

    return phone_list

def inter_pitch(y,y_flag):
    frame_num = y.shape[0]
    i = 0
    last = -1
    while(i<frame_num):
        if y_flag[i] == 0:
            while True:
                if y_flag[i]==0:
                    if i == frame_num-1:
                        if last !=-1:
                            y[last+1:] = y[last]
                        i+=1
                        break
                    i+=1
                else:
                    break
            if i >= frame_num:
                break
            elif last == -1:
                y[:i] = y[i]
            else:
                inter_num = i-last+1
                fy = np.array([y[last],y[i]])
                fx = np.linspace(0, 1, num=2)
                f = interp1d(fx,fy)
                fx_new = np.linspace(0,1,inter_num)
                fy_new = f(fx_new)
                y[last+1:i] = fy_new[1:-1]
                last = i
                i+=1

        else:
            last = i
            i+=1
    return y

def get_audio_feature_from_audio(audio_path):
    sample_rate, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        if np.min(audio[:, 0]) <= 0:
            audio = audio[:, 1]
        else:
            audio = audio[:, 0]

    audio = audio - np.mean(audio)
    audio = audio / np.max(np.abs(audio))
    a = python_speech_features.mfcc(audio, sample_rate)
    b = python_speech_features.logfbank(audio, sample_rate)
    c, _ = pyworld.harvest(audio, sample_rate, frame_period=10)
    c_flag = (c == 0.0) ^ 1
    c = inter_pitch(c, c_flag)
    c = np.expand_dims(c, axis=1)
    c_flag = np.expand_dims(c_flag, axis=1)
    frame_num = np.min([a.shape[0], b.shape[0], c.shape[0]])

    cat = np.concatenate([a[:frame_num], b[:frame_num], c[:frame_num], c_flag[:frame_num]], axis=1)
    return cat

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """
    # def __init__(self, return_wave=False):
        
    def __call__(self, batch):
        # for k in batch[0].keys():
        #     print(k)
        #     print(k, batch[0][k].shape)
        batch[0]['source'] = torch.from_numpy(batch[0]['source']).unsqueeze(0)
        return batch[0]

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]

if __name__ == "__main__":
    import os, sys
    import yaml
    import time
    from argparse import ArgumentParser
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="config/qvt_img_pca_sync_2.yaml", help="path to config")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = FramesWavsDatasetMEL25VoxBoxQG2ImgAll(is_train=True, **config['dataset_params'])
    collate_fn = Collater()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16, drop_last=True, collate_fn=collate_fn)
    s = time.time()
    for x in tqdm(dataloader):
        continue
    d = time.time()
    print(d-s)