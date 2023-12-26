from os.path import basename, exists
import os
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
from skimage import io, img_as_float32
import numpy as np
import torch

import face_detection

fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda')

def detect_bbox(img_names):
    bboxs = []
    for img_name in img_names:
        img = img_as_float32(io.imread(img_name)).transpose((2, 0, 1))
        img = np.transpose(img[np.newaxis], (0,2,3,1))[...,::-1]
        bbox = fa.get_detections_for_batch(img*255)
        if bbox is not None:
            bboxs.append(bbox[0])
        else:
            bboxs.append(None)
    assert(len(bboxs)==len(img_names))
    return bboxs

def main(args):
    # file_images = glob('/data5/gy/mead/images_evp_25/t*/*')
    # file_images = glob('/data/gy/vox/voxs_images/*')
    # file_images = glob('/data/gy/vox/voxs_images/*')
    file_images = glob('/data2/gy/lrw/lrw_images/*')
    file_images.sort()
    #file_images = file_images[6669*24:6669*25]
    p = args.part
    # t = len(file_images)//9 + 1
    t = len(file_images)
    #f = open('bboxs_extract.txt', 'w')
    for fi in tqdm(file_images[t*p:t*(p+1)]):
        out = basename(fi)
        # outpath =f'./bboxs/{out}.npy's
        outpath =f'/data2/gy/lrw/lrw_bbox/{out}.npy'
        if exists(outpath):
            if exists(outpath):
                try:
                    np.load(outpath, allow_pickle=True)
                    continue
                except:
                    # f.write(out+'\n')
                    print(outpath)
                    
        #else:
            #f.write(out+'\n')
        images = glob(fi+'/*.jpg')
        images.sort()
        bboxs = detect_bbox(images)
        np.save(outpath, bboxs)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--files", default="*", help="filenames")
    parser.add_argument("--part", default="0", type=int, help="part")
    args = parser.parse_args()
    main(args)
