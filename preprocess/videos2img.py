import cv2
import os
import glob
from tqdm import tqdm

from argparse import ArgumentParser

def video2img(path):
    vc = cv2.VideoCapture(path) #读入视频文件
    rval=vc.isOpened()
    outputpath = path.replace('cropped','imgs')[:-4]
    #outputpath = path[:-4]
    c=0
    if not rval:
        f = open('./failed.txt', 'a+')
        f.write(path+'\n')
        f.close()
    while rval:   #循环读取视频帧
        c = c + 1
        rval, frame = vc.read()
        if c==1:
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)
            else:
                if (vc.get(cv2.CAP_PROP_FRAME_COUNT)==len(glob.glob(outputpath+'/*.jpg'))):
                    break
        if rval:
            cv2.imwrite(outputpath + '/' + str(c).zfill(4) + '.jpg', frame) #存储为图像
        else:
            break
    vc.release()

def main(args):
    videos = glob.glob(f'./cropped/{args.files}.mp4')
    videos.sort()
    for v in tqdm(videos):
        video2img(v)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--files", default="*", help="filenames")
    parser.add_argument("--part", default="0", type=int, help="part")
    args = parser.parse_args()
    main(args)
