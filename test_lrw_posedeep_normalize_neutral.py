import os
import numpy as np
import torch
import yaml
from modules.generator import OcclusionAwareSPADEGeneratorEam
from modules.keypoint_detector import KPDetector
import argparse
import imageio
from modules.transformer import Audio2kpTransformerBBoxQDeepPrompt as Audio2kpTransformer
from modules.prompt import EmotionDeepPrompt, EmotionalDeformationTransformer
from scipy.io import wavfile

# from frames_dataset_transformer25 import parse_phoneme_file

from modules.model_transformer import get_rotation_matrix, keypoint_transformation
from skimage import io, img_as_float32
import torchaudio
import soundfile as sf
from scipy.spatial import ConvexHull

import torch.nn.functional as F
import glob
from tqdm import tqdm
import gzip

from animate import normalize_kp

emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']
emo_label_full = ['angry',  'contempt',  'disgusted',  'fear',  'happy',  'neutral',  'sad',  'surprised']
latent_dim = 16

MEL_PARAMS_25 = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 640,
    "hop_length": 640
}

to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS_25)
mean, std = -4, 4

expU = torch.from_numpy(np.load('./expPCAnorm_fin/U_mead.npy')[:,:32])
expmean = torch.from_numpy(np.load('./expPCAnorm_fin/mean_mead.npy'))

root_lrw = '/data2/gy/lrw'

# def normalize_kp(kp_source, kp_driving, kp_driving_initial,
#                  use_relative_movement=True, use_relative_jacobian=True):

#     kp_new = {k: v for k, v in kp_driving.items()}
#     if use_relative_movement:
#         kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
#         # kp_value_diff *= adapt_movement_scale
#         kp_new['value'] = kp_value_diff + kp_source['value']

#         if use_relative_jacobian:
#             jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
#             kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

#     return kp_new

def _load_tensor(data):
    wave_path = data
    wave, sr = sf.read(wave_path)
    wave_tensor = torch.from_numpy(wave).float()
    return wave_tensor

def build_model(config, device_ids=[0]):
    generator = OcclusionAwareSPADEGeneratorEam(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])
    if torch.cuda.is_available():
        print('cuda is available')
        generator.to(device_ids[0])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(device_ids[0])


    audio2kptransformer = Audio2kpTransformer(**config['model_params']['audio2kp_params'], face_ea=True)

    if torch.cuda.is_available():
        audio2kptransformer.to(device_ids[0])
    
    sidetuning = EmotionalDeformationTransformer(**config['model_params']['audio2kp_params'])

    if torch.cuda.is_available():
        sidetuning.to(device_ids[0])

    emotionprompt = EmotionDeepPrompt()

    if torch.cuda.is_available():
        emotionprompt.to(device_ids[0])

    return generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt



def prepare_test_data(img_path, audio_path, opt):
    sr,_ = wavfile.read(audio_path)
    temp_audio = audio_path

    # print(img_path, audio_path)
    source_latent = np.load(img_path.replace('images', 'latent')[:-9]+'.npy', allow_pickle=True)
    # isp = img_path.split('/')
    he_source = {}
    for k in source_latent[1].keys():
        he_source[k] = torch.from_numpy(source_latent[1][k][0]).unsqueeze(0).cuda()
    
    # source images
    source_img = img_as_float32(io.imread(img_path)).transpose((2, 0, 1))
    asp = os.path.basename(audio_path)[:-4]
    
    # driving latent
    latent_path_driving = f'{root_lrw}/lrw_latent/{asp}.npy'
    pose_gz = gzip.GzipFile(f'{root_lrw}/poseimg/{asp}.npy.gz', 'r')
    poseimg = np.load(pose_gz)
    deepfeature = np.load(f'{root_lrw}/lrw_df32/{asp}.npy')
    driving_latent = np.load(latent_path_driving[:-4]+'.npy', allow_pickle=True)
    he_driving = driving_latent[1]

    # latent code
    y_trg = emo_label.index('neu')
    z_trg = torch.randn(latent_dim)

    # gt frame number
    path = latent_path_driving[:-4].replace('latent', 'images')
    frames = glob.glob(path+'/*.jpg')
    num_frames = len(frames)
    # print(num_frames,he_driving['exp'], latent_path_driving)

    wave_tensor = _load_tensor(audio_path)
    if len(wave_tensor.shape) > 1:
        wave_tensor = wave_tensor[:, 0]
    # print(wave_tensor[:100])
    # print(to_melspec(torch.zeros_like(wave_tensor)+ -6.1035e-05))
    mel_tensor = to_melspec(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor) - mean) / std
    name_len = min(mel_tensor.shape[1], deepfeature.shape[0], poseimg.shape[0])

    # audio_seq = audio_feature#[40:]
    # ph_seq = phs["phone_list"]

    audio_frames = []
    poseimgs = []
    deep_feature = []
    
    pad = torch.zeros((mel_tensor.shape[0]))
    deep_pad = np.zeros([deepfeature.shape[1], deepfeature.shape[2]])

    for rid in range(0, num_frames):
        audio = []
        poses = []
        deeps = []
        for i in range(rid - opt['num_w'], rid + opt['num_w'] + 1):
            if i < 0:
                audio.append(pad)
                poses.append(poseimg[0])
                deeps.append(deep_pad)
            elif i >= name_len:
                audio.append(pad)
                poses.append(poseimg[-1])
                deeps.append(deep_pad)
            else:
                audio.append(mel_tensor[:, i])
                poses.append(poseimg[i])
                deeps.append(deepfeature[i])

        audio_frames.append(torch.stack(audio, dim=1))
        poseimgs.append(poses)
        deep_feature.append(deeps)
    audio_frames = torch.stack(audio_frames, dim=0)
    poseimgs = torch.from_numpy(np.array(poseimgs))
    deep_feature = torch.from_numpy(np.array(deep_feature)).to(torch.float)
    # print(audio_frames.shape) # len, 80, 11
    return audio_frames, poseimgs, deep_feature, source_img, he_source, he_driving, num_frames,  y_trg, z_trg, latent_path_driving

def load_ckpt(ckpt, kp_detector, generator, audio2kptransformer):
    checkpoint = torch.load(ckpt)
    if audio2kptransformer is not None:
        audio2kptransformer.load_state_dict(checkpoint['audio2kptransformer'])
    if generator is not None:
        generator.load_state_dict(checkpoint['generator'])
    if kp_detector is not None:
        kp_detector.load_state_dict(checkpoint['kp_detector'])

gt_i = [5,6,7,8]
def test_mead(ckpt, part=0, save_dir=" "):
    with open("config/deepprompt_eam3d_st_tanh_304_3090_all.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # temp_audio = audio_path
    # print(audio_path)
    cur_path = os.getcwd()
    generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt = build_model(config)
    load_ckpt(ckpt, kp_detector=kp_detector, generator=generator, audio2kptransformer=audio2kptransformer)
    
    audio2kptransformer.eval()
    generator.eval()
    kp_detector.eval()
    sidetuning.eval()
    emotionprompt.eval()

    all_wavs2 = []
    all_wavs = glob.glob(f'{root_lrw}/lrw_wavs/*.wav')
    # name = np.load('../ost/random_filter100.npy')
    all_wavs2 = all_wavs

    # for i in all_wavs:
        # if os.path.basename(i)[:-4] in name:
            # all_wavs2.append(i)
    all_wavs2.sort()
    loss_latents = []
    loss_pca_emos = []
    loss_y = []
    
    t = len(all_wavs2)//4 + 1
    all_wavs2 = all_wavs2[t*part: t*(part+1)]
    

    for ind in tqdm(range(len(all_wavs2))):
        audio_path = all_wavs2[ind]
        name_a = os.path.basename(audio_path)[:-4]
        img_path = f'{root_lrw}/lrw_images/{name_a}/0001.jpg'

        # read in data
        audio_frames, poseimgs, deep_feature, source_img, he_source, he_driving, num_frames, y_trg, z_trg, latent_path_driving = prepare_test_data(img_path, audio_path, config['model_params']['audio2kp_params'])


        with torch.no_grad():
            source_img = torch.from_numpy(source_img).unsqueeze(0).cuda()
            kp_canonical = kp_detector(source_img, with_feature=True)     # {'value': value, 'jacobian': jacobian}   
            kp_cano = kp_canonical['value']

            x = {}
            # x['pho'] = ph_frames.unsqueeze(0).cuda()
            x['mel'] = audio_frames.unsqueeze(1).unsqueeze(0).cuda()
            x['z_trg'] = z_trg.unsqueeze(0).cuda()
            x['y_trg'] = torch.tensor(y_trg, dtype=torch.long).cuda().reshape(1)
            x['pose'] = poseimgs.cuda()
            x['deep'] = deep_feature.cuda().unsqueeze(0)
            x['he_driving'] = {'yaw': torch.from_numpy(he_driving['yaw']).cuda().unsqueeze(0), 
                            'pitch': torch.from_numpy(he_driving['pitch']).cuda().unsqueeze(0), 
                            'roll': torch.from_numpy(he_driving['roll']).cuda().unsqueeze(0), 
                            't': torch.from_numpy(he_driving['t']).cuda().unsqueeze(0), 
                            }
            ### emotion prompt
            emoprompt, deepprompt = emotionprompt(x)
            
            he_driving_emo, input_st = audio2kptransformer(x, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           # {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}
            emo_exps = sidetuning(input_st, emoprompt, deepprompt).reshape(-1, 45)     
            exp = he_driving_emo['emo']
            device = exp.get_device()
            exp = torch.mm(exp, expU.t().to(device))
            exp = exp + expmean.expand_as(exp).to(device)

            exp = exp + emo_exps

            source_area = ConvexHull(kp_cano[0].cpu().numpy()).volume
            exp = exp * source_area

            # print(he_driving['yaw'].shape) # len 66
            he_new_driving = {'yaw': torch.from_numpy(he_driving['yaw']).cuda(), 
                            'pitch': torch.from_numpy(he_driving['pitch']).cuda(), 
                            'roll': torch.from_numpy(he_driving['roll']).cuda(), 
                            't': torch.from_numpy(he_driving['t']).cuda(), 
                            #   'exp': torch.from_numpy(he_driving['exp']).cuda()}
                            'exp': exp}
            he_driving['exp'] = torch.from_numpy(he_driving['exp']).cuda()
            # print(he_new_driving['exp'][:, 0])
            loss_latent = F.mse_loss(he_new_driving['exp'], he_driving['exp'])
            pca_exp = torch.mm(he_driving['exp'].squeeze(0)/source_area - expmean.expand_as(he_driving['exp'].squeeze(0)).to(device), expU.to(device))
            loss_pca_emo = F.mse_loss(he_driving_emo['emo'], pca_exp)
            new_exp = he_new_driving['exp'].reshape(-1,15,3)
            gt_exp = he_driving['exp'].reshape(-1,15,3)
            for i in gt_i:
                loss_y.append(torch.abs(new_exp[:, i, 1] - gt_exp[:, i, 1]).mean().cpu().numpy())
            loss_latents.append(loss_latent.cpu().numpy())
            loss_pca_emos.append(loss_pca_emo.cpu().numpy())
            # print(kp_canonical['value'].shape)
            kp_source = keypoint_transformation(kp_canonical, he_source, False)
            kp_driving = keypoint_transformation(kp_canonical, he_new_driving, False)
            # kp_source['value'] = kp_source['value'].expand_as(kp_driving['value'])
            bs = kp_source['value'].shape[0]
            # print(kp_driving['value'].shape)
            
            drive_first = {}
            drive_first['value'] = kp_driving['value'][0].unsqueeze(0)
           
            normalized = normalize_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=drive_first,\
                                        adapt_movement_scale=True, use_relative_movement=True)

            predictions_gen = []
            for i in range(num_frames):
                kp_si = {}
                kp_si['value'] = kp_source['value'][0].unsqueeze(0)
                kp_di = {}
                kp_di['value'] = normalized['value'][i].unsqueeze(0)
                # kp_di['value'] = kp_driving['value'][i].unsqueeze(0)
                # print(kp_source['value'].shape)
                # print(kp_driving['value'].shape)
                # assert(0)
                generated = generator(source_img, kp_source=kp_si, kp_driving=kp_di, prompt=emoprompt)
                # print(generated['prediction'])
                predictions_gen.append(
                    (np.transpose(generated['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

        log_dir = save_dir
        os.makedirs(os.path.join(log_dir, "temp"), exist_ok=True)

        f_name = os.path.basename(img_path[:-9]) + ".mp4"
        # kwargs = {'duration': 1. / 25.0}
        video_path = os.path.join(log_dir, "temp", f_name)
        # print("save video to: ", video_path)
        imageio.mimsave(video_path, predictions_gen, fps=25.0)

        # audio_path = os.path.join(audio_dir, x['name'][0].replace(".mp4", ".wav"))
        save_video = os.path.join(log_dir, f_name)
        cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
        os.system(cmd)
        os.remove(video_path)
    
    print('mean loss_latents', np.mean(loss_latents))
    print('mean loss_pca_emos', np.mean(loss_pca_emos))
    print('mean loss_y ', np.mean(loss_y))






if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_dir", type=str, default=" ", help="path of the output video")
    argparser.add_argument("--part", default=0, type=int, help="part wavs")
    argparser.add_argument("--name", type=str, default=" ", help="path of the output video")
    argparser.add_argument("--mode", type=int, default=0, help="test mode 0: a ckpt 1: a ckpt file")
    args = argparser.parse_args()

    # name = 'vt2mel25_2_vox_head_587'
    # name = 'a2kp_pretrain_496_2'
    # name = 'qvt_528'
    # name = 'qvt_img_538'
    name = 'a2kp_posedeep_img_synconly_mead_479'


    if len (args.name) > 1:
        name = args.name
        print(name)
    if args.mode == 0:
        test_mead(f'./ckpt/{name}.pth.tar', args.part, save_dir=f'./result_lrw/{name}_lrw_norm/')
    elif args.mode == 1:
        ckpt_paths = glob.glob(f'../ost/output/{name}/*.pth.tar')
        ckpt_paths.sort()
        for ckpt in ckpt_paths:
            epoch = int(os.path.basename(ckpt)[:8])
            #if epoch %4==1:
            if epoch == 313:
                nsp = name.split(' ')
                savefile = nsp[0]+'_'+nsp[1][:8]
                # ckpt = ckpt.replace(' ', '\\ ')
                print(f'./{savefile}_{epoch}/')
                test_mead(f'./{ckpt}', args.part, save_dir=f'./result_lrw/{savefile}_{epoch}_lrw_norm_25k_1202/')
        
