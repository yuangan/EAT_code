from glob import glob
import os

allmp4s = glob('./video/*.mp4')
path_fps25='./video_fps25'
os.makedirs(path_fps25, exist_ok=True)

for mp4 in allmp4s:
    name = os.path.basename(mp4)
    os.system(f'ffmpeg -y -i {mp4} -filter:v fps=25 -ac 1 -ar 16000 -crf 10 {path_fps25}/{name}')
    os.system(f'ffmpeg -y -i {path_fps25}/{name} {path_fps25}/{name[:-4]}.wav')

#============== extract lmk for crop =================
print('============== extract lmk for crop =================')
os.system(f'python extract_lmks_eat.py {path_fps25}')

#======= extract speech in deepspeech_features =======
print('======= extract speech in deepspeech_features =======')
os.chdir('./deepspeech_features/')
os.system(f'python extract_ds_features.py --input=../{path_fps25}')
os.chdir('../')
os.system('python deepfeature32.py')

#=================== crop videos =====================
print('=================== crop videos =====================')
os.chdir('./vid2vid/')
os.system('python data_preprocess.py --dataset_mode preprocess_eat')
os.chdir('../')

#========== extract latent from cropped videos =======
print('========== extract latent from cropped videos =======')
os.system('python videos2img.py')
os.system('python latent_extractor.py')

#=========== extract poseimg from latent =============
print('=========== extract poseimg from latent =============')
os.system('python generate_poseimg.py')

#============== organize file for demo ===============
print('============== organize file for demo ===============')
for mp4 in allmp4s:
    name = os.path.basename(mp4)[:-4]
    filename=f'../demo/video_processed/{name}'
    os.makedirs(f'{filename}/deepfeature32', exist_ok=True)
    os.makedirs(f'{filename}/latent_evp_25', exist_ok=True)
    os.makedirs(f'{filename}/poseimg', exist_ok=True)
    os.makedirs(f'{filename}/images_evp_25/cropped', exist_ok=True)
    # wav
    os.system(f'cp ./video_fps25/{name}.wav {filename}')
    # deepfeature32
    os.system(f'cp ./deepfeature32/{name}.npy {filename}/deepfeature32/')
    # latent
    os.system(f'cp ./latents/{name}.npy {filename}/latent_evp_25/')
    # poseimg
    os.system(f'cp ./poseimg/{name}.npy.gz {filename}/poseimg/')
    # images_evp_25
    os.system(f'cp ./imgs/{name}/* {filename}/images_evp_25/cropped/')

