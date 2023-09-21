import torch
import os, sys
sys.path.append('../')
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.keypoint_detector import KPDetector, HEEstimator
from scipy.spatial import ConvexHull
import glob
import time


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    if not cpu:
        he_estimator.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    he_estimator.load_state_dict(checkpoint['he_estimator'])
    
    if not cpu:
        kp_detector = DataParallelWithCallback(kp_detector)
        he_estimator = DataParallelWithCallback(he_estimator)

    kp_detector.eval()
    he_estimator.eval()
    
    return kp_detector, he_estimator

def estimate_latent(driving_video, kp_detector, he_estimator):
    with torch.no_grad():
        predictions = []
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
        kp_canonical = kp_detector(driving[:, :, 0])
        he_drivings = {'yaw': [], 'pitch': [], 'roll': [], 't': [], 'exp': []}

        for frame_idx in range(driving.shape[2]):
            driving_frame = driving[:, :, frame_idx]
            he_driving = he_estimator(driving_frame)
            for k in he_drivings.keys():
                he_drivings[k].append(he_driving[k])
    return kp_canonical, he_drivings

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='../config/vox-256-spade.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='../ckpt/pretrain_new_274.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")
    parser.add_argument("--part", default=0, type=int, help="part emotion")

    opt = parser.parse_args()
    part = opt.part
    trainlist = glob.glob('./imgs/*')
    trainlist.sort()
    kp_detector, he_estimator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)
    if not os.path.exists('./latents/'):
        os.makedirs('./latents')
    #     os.makedirs('./output/latent_evp/test/')
    #for videoname in tqdm(trainlist[part*2850 : (part+1)*2850]):
    for videoname in tqdm(trainlist):
        path_frames = glob.glob(videoname+'/*.jpg')
        path_frames.sort()
        driving_frames = []
        for im in path_frames:
            driving_frames.append(imageio.imread(im))
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_frames]

        kc, he = estimate_latent(driving_video, kp_detector, he_estimator)
        kc = kc['value'].cpu().numpy()
        for k in he:
            he[k] = torch.cat(he[k]).cpu().numpy()
        np.save('./latents/'+os.path.basename(videoname), [kc, he])
    print('=============done==============')
