import glob
import numpy as np
import os
from os.path import basename,exists

import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import sys
import cv2
sys.path.append('../')
from modules.util import AntiAliasInterpolation2d
import gzip

down_pose = AntiAliasInterpolation2d(1,0.25).cuda()

def headpose_pred_to_degree(pred):
    pred = torch.from_numpy(pred).cuda()
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

def get_pose_img(he_driving):
    yaw = headpose_pred_to_degree(he_driving['yaw'])
    pitch = headpose_pred_to_degree(he_driving['pitch'])
    roll = headpose_pred_to_degree(he_driving['roll'])
    rot = get_rotation_matrix(yaw, pitch, roll).cpu().numpy().astype(np.double)
    t = he_driving['t'].astype(np.double)
    poseimgs = []
    for i in range(rot.shape[0]):
        ri = rot[i]
        ti = t[i]
        img = np.zeros([256, 256])
        draw_annotation_box(img, ri, ti)
        poseimgs.append(img)
    poseimgs = torch.from_numpy(np.array(poseimgs))
    down_poseimgs = down_pose(poseimgs.unsqueeze(1).cuda().to(torch.float))
    return down_poseimgs


def main(args):
    all_latents = glob.glob('./latents/*.npy')
    all_latents.sort()
    os.makedirs('./poseimg', exist_ok=True)
    for lat in tqdm(all_latents):
        out = basename(lat)[:-4]
        outpath =f'./poseimg/{out}.npy.gz'
        if exists(outpath):
            try:
                f = gzip.GzipFile(f'{outpath}', "r")
                np.load(f)
                continue
            except:
                print(outpath)
        kp_cano, he_driving = np.load(lat, allow_pickle=True)
        poseimgs = get_pose_img(he_driving)
        f = gzip.GzipFile(f'{outpath}', "w")
        np.save(file=f, arr=poseimgs.cpu().numpy())

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--files", default="*", help="filenames")
    parser.add_argument("--part", default="0", type=int, help="part")
    args = parser.parse_args()
    main(args)




