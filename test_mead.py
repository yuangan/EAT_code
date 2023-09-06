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

from modules.model_transformer import keypoint_transformation
from skimage import io, img_as_float32
import torchaudio
import soundfile as sf
from scipy.spatial import ConvexHull

import torch.nn.functional as F
import glob
from tqdm import tqdm
import gzip

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

root_mead = './mead_data'
def normalize_kp(kp_source, kp_driving, kp_driving_initial,
                 use_relative_movement=True, use_relative_jacobian=True):

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

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

    source_latent = np.load(img_path.replace('images', 'latent')[:-9]+'.npy', allow_pickle=True)
    he_source = {}
    for k in source_latent[1].keys():
        he_source[k] = torch.from_numpy(source_latent[1][k][0]).unsqueeze(0).cuda()
    
    # source images
    source_img = img_as_float32(io.imread(img_path)).transpose((2, 0, 1))
    asp = os.path.basename(audio_path)[:-4].split('_')
    
    # latent code
    y_trg = emo_label.index(asp[1])
    z_trg = torch.randn(latent_dim)
    
    # driving latent
    latent_path_driving = f'{root_mead}/latent_evp_25/test/{asp[0]}_{asp[1][:3]}_{asp[2][-1]}_{asp[3]}.npy'
    pose_gz = gzip.GzipFile(f'{root_mead}/poseimg/{asp[0]}_{asp[1][:3]}_{asp[2][-1]}_{asp[3]}.npy.gz', 'r')
    poseimg = np.load(pose_gz)
    deepfeature = np.load(f'{root_mead}/deepfeature32/{asp[0]}_{asp[1][:3]}_{asp[2][-1]}_{asp[3]}.npy')
    driving_latent = np.load(latent_path_driving[:-4]+'.npy', allow_pickle=True)
    he_driving = driving_latent[1]

    # gt frame number
    path = latent_path_driving[:-4].replace('latent', 'images')
    frames = glob.glob(path+'/*.jpg')
    num_frames = len(frames)

    wave_tensor = _load_tensor(audio_path)
    if len(wave_tensor.shape) > 1:
        wave_tensor = wave_tensor[:, 0]
    mel_tensor = to_melspec(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor) - mean) / std
    name_len = min(mel_tensor.shape[1], deepfeature.shape[0], poseimg.shape[0])

    audio_frames = []
    poseimgs = []
    deep_feature = []
    
    pad, deep_pad = np.load('pad.npy', allow_pickle=True)

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
    return audio_frames, poseimgs, deep_feature, source_img, he_source, he_driving, num_frames, y_trg, z_trg, latent_path_driving

def load_ckpt(ckpt, kp_detector, generator, audio2kptransformer, sidetuning, emotionprompt):
    checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
    if audio2kptransformer is not None:
        audio2kptransformer.load_state_dict(checkpoint['audio2kptransformer'])
    if generator is not None:
        generator.load_state_dict(checkpoint['generator'])
    if kp_detector is not None:
        kp_detector.load_state_dict(checkpoint['kp_detector'])
    if sidetuning is not None:
        sidetuning.load_state_dict(checkpoint['sidetuning'])
    if emotionprompt is not None:
        emotionprompt.load_state_dict(checkpoint['emotionprompt'])

def test_mead(ckpt, part=0, mode='sample', save_dir=" "):
    with open("config/deepprompt_eam3d_st_tanh_304_3090_all.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    cur_path = os.getcwd()
    print('========='*2)
    print('Load ckpt.')
    generator, kp_detector, audio2kptransformer, sidetuning, emotionprompt = build_model(config)
    load_ckpt(ckpt, kp_detector=kp_detector, generator=generator, audio2kptransformer=audio2kptransformer, sidetuning=sidetuning, emotionprompt=emotionprompt)
    
    audio2kptransformer.eval()
    generator.eval()
    kp_detector.eval()
    sidetuning.eval()
    emotionprompt.eval()

    ids = ['W015', 'M003', 'M030', 'W009']
    all_wavs = []
    all_wavs2 = []
    print('========='*2)
    print('Start Test ', ids[part], '.')
    for i in [ids[part]]:
        all_wavs.extend(glob.glob(f'{root_mead}/wav_16000/{i}*'))
    name = np.load('./rand_sample_ours_mead100.npy')

    if mode=='sample':
        for i in all_wavs:
            if ids[part]+'_'+os.path.basename(i)[:-4] in name:
                all_wavs2.append(i)
    else:
        all_wavs2=all_wavs 

    all_wavs2.sort()

    loss_latents = []
    loss_pca_emos = []
    loss_y = []

    for ind in tqdm(range(len(all_wavs2))):
        iid = ids[part]
        img_path = glob.glob(f'{root_mead}/images_evp_25/new_neutral_2/{iid}/{iid}.*g')[0] # png/jpg
        audio_path = all_wavs2[ind]

        # read in data
        audio_frames, poseimgs, deep_feature, source_img, he_source, he_driving, num_frames, y_trg, z_trg, latent_path_driving = prepare_test_data(img_path, audio_path, config['model_params']['audio2kp_params'])


        with torch.no_grad():
            source_img = torch.from_numpy(source_img).unsqueeze(0).cuda()
            kp_canonical = kp_detector(source_img, with_feature=True) 
            kp_cano = kp_canonical['value']

            x = {}
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
            a2kp_exps = []
            emo_exps = []
            T = 5
            if T == 1:
                for i in range(x['mel'].shape[1]):
                    xi = {}
                    xi['mel'] = x['mel'][:,i,:,:,:].unsqueeze(1)
                    xi['z_trg'] = x['z_trg']
                    xi['y_trg'] = x['y_trg']
                    xi['pose'] = x['pose'][i,:,:,:,:].unsqueeze(0)
                    xi['deep'] = x['deep'][:,i,:,:,:].unsqueeze(1)
                    xi['he_driving'] = {'yaw': x['he_driving']['yaw'][:,i,:].unsqueeze(0), 
                                'pitch': x['he_driving']['pitch'][:,i,:].unsqueeze(0), 
                                'roll': x['he_driving']['roll'][:,i,:].unsqueeze(0), 
                                't': x['he_driving']['t'][:,i,:].unsqueeze(0), 
                                }
                    he_driving_emo_xi, input_st_xi = audio2kptransformer(xi, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)
                    emo_exp = sidetuning(input_st_xi, emoprompt, deepprompt)
                    a2kp_exps.append(he_driving_emo_xi['emo'])
                    emo_exps.append(emo_exp)
            elif T is not None:
                for i in range(x['mel'].shape[1]//T+1):
                    if i*T >= x['mel'].shape[1]:
                        break
                    xi = {}
                    xi['mel'] = x['mel'][:,i*T:(i+1)*T,:,:,:]
                    xi['z_trg'] = x['z_trg']
                    xi['y_trg'] = x['y_trg']
                    xi['pose'] = x['pose'][i*T:(i+1)*T,:,:,:,:]
                    xi['deep'] = x['deep'][:,i*T:(i+1)*T,:,:,:]
                    xi['he_driving'] = {'yaw': x['he_driving']['yaw'][:,i*T:(i+1)*T,:], 
                                'pitch': x['he_driving']['pitch'][:,i*T:(i+1)*T,:], 
                                'roll': x['he_driving']['roll'][:,i*T:(i+1)*T,:], 
                                't': x['he_driving']['t'][:,i*T:(i+1)*T,:], 
                                }
                    he_driving_emo_xi, input_st_xi = audio2kptransformer(xi, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           
                    emo_exp = sidetuning(input_st_xi, emoprompt, deepprompt)
                    a2kp_exps.append(he_driving_emo_xi['emo'])
                    emo_exps.append(emo_exp)
            
            if T is None:
                ### test all frames in a batch, require large memory
                he_driving_emo, input_st = audio2kptransformer(x, kp_canonical, emoprompt=emoprompt, deepprompt=deepprompt, side=True)           
                emo_exps = sidetuning(input_st, emoprompt, deepprompt).reshape(-1, 45)
            else:
                he_driving_emo = {}
                he_driving_emo['emo'] = torch.cat(a2kp_exps, dim=0)
                emo_exps = torch.cat(emo_exps, dim=0).reshape(-1, 45)

            exp = he_driving_emo['emo']
            device = exp.get_device()
            exp = torch.mm(exp, expU.t().to(device))
            exp = exp + expmean.expand_as(exp).to(device)
            exp = exp + emo_exps # add emotional deformation expression to A2ET expression

            source_area = ConvexHull(kp_cano[0].cpu().numpy()).volume
            exp = exp * source_area

            he_new_driving = {'yaw': torch.from_numpy(he_driving['yaw']).cuda(), 
                            'pitch': torch.from_numpy(he_driving['pitch']).cuda(), 
                            'roll': torch.from_numpy(he_driving['roll']).cuda(), 
                            't': torch.from_numpy(he_driving['t']).cuda(), 
                            'exp': exp}
            he_driving['exp'] = torch.from_numpy(he_driving['exp']).cuda() # used in calculating loss only
            loss_latent = F.mse_loss(he_new_driving['exp'], he_driving['exp'])
            pca_exp = torch.mm(he_driving['exp'].squeeze(0)/source_area - expmean.expand_as(he_driving['exp'].squeeze(0)).to(device), expU.to(device))
            loss_pca_emo = F.mse_loss(he_driving_emo['emo'], pca_exp)
            loss_latents.append(loss_latent.cpu().numpy())
            loss_pca_emos.append(loss_pca_emo.cpu().numpy())

            kp_source = keypoint_transformation(kp_canonical, he_source, False)
            mean_source = torch.mean(kp_source['value'], dim=1)[0]
            kp_driving = keypoint_transformation(kp_canonical, he_new_driving, False)
            mean_driving = torch.mean(torch.mean(kp_driving['value'], dim=1), dim=0)
            kp_driving['value'] = kp_driving['value']+(mean_source-mean_driving).unsqueeze(0).unsqueeze(0)
            bs = kp_source['value'].shape[0]
            predictions_gen = []
            for i in range(num_frames):
                kp_si = {}
                kp_si['value'] = kp_source['value'][0].unsqueeze(0)
                kp_di = {}
                kp_di['value'] = kp_driving['value'][i].unsqueeze(0)
                generated = generator(source_img, kp_source=kp_si, kp_driving=kp_di, prompt=emoprompt)
                predictions_gen.append(
                    (np.transpose(generated['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

        log_dir = save_dir
        os.makedirs(os.path.join(log_dir, "temp"), exist_ok=True)

        f_name = os.path.basename(img_path[:-9]) + "_" + os.path.basename(latent_path_driving)[:-4] + ".mp4"
        video_path = os.path.join(log_dir, "temp", f_name)
        imageio.mimsave(video_path, predictions_gen, fps=25.0)

        save_video = os.path.join(log_dir, f_name)
        cmd = r'ffmpeg -loglevel error -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
        os.system(cmd)
    print('---------'*2)
    print('mean loss_latents', np.mean(loss_latents))
    print('mean loss_pca_emos', np.mean(loss_pca_emos))
    print('========='*2)
    print('Test Success! Results are in ', save_dir)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--save_dir", type=str, default=" ", help="path of the output video")
    argparser.add_argument("--part", default=0, type=int, help="part wavs")
    argparser.add_argument("--mode", default=0, type=int, help="0: quick mode (sample 100 videos) 1: full test")
    argparser.add_argument("--name", type=str, default="deepprompt_eam3d_all_final_313", help="path of the output video")
    args = argparser.parse_args()

    if args.mode==0:
        mode='sample'
    else:
        mode='full'
    if len(args.name) > 1:
        name = args.name
        print(name)
    test_mead(f'./ckpt/{name}.pth.tar', args.part, mode=mode, save_dir=f'./mead_result/{name}/')
    
