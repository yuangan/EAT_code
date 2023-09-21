from cmath import inf
from turtle import distance
import cv2
import dlib
import numpy as np
from imutils import face_utils
import math
import os
import glob
import json
from tqdm import tqdm
import sys

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../demo/shape_predictor_68_face_landmarks.dat')

path_fps25=sys.argv[1]
vpths = glob.glob(f'{path_fps25}/*.mp4')
sav_rt = './lmk_fps25/'
os.makedirs(sav_rt,exist_ok=True)

log_f = open('extract_lmks_fps25.txt','a+')

for vpth in tqdm(vpths):
    vsp  = vpth.split('/')
    sav_n = '{}/{}.json'.format( sav_rt , vsp[-1].split('.')[0] )

    if os.path.exists(sav_n): continue

    vreader = cv2.VideoCapture(vpth)
        
    shape = None
    fid = -1
    while True:
        ret , img = vreader.read()
        if not ret: break
        fid += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            break
        if shape is not None: 
            ann_img = img
            for p in shape: 
                cv2.circle(ann_img,p,1,(255,0,0),1)
            cv2.imwrite( sav_n.replace('.json','.jpg') , ann_img )
            break
        # assert(0)
    
    if shape is None : 
        print(vpth)
        log_f.writelines('{}\n'.format(vpth))
        continue

    json.dump( [fid] + shape.tolist() , open(sav_n,'w') )







