<div align="center">

# Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation (EAT <a href="https://github.com/yuangan/EAT_code"><img src="./doc/favicon_eat.png" style="width: 25px;"></a>)

<a href="https://yuangan.github.io/"><strong>Yuan Gan</strong></a>
·
<a href="https://z-x-yang.github.io/"><strong>Zongxin Yang</strong></a>
·
<a><strong>Xihang Yue</strong></a>
·
<a href="https://scholar.google.com/citations?user=zzW8d-wAAAAJ&hl=zh-CN&oi=ao"><strong>Lingyun Sun</strong></a>
·
<a href="https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en"><strong>Yi Yang</strong></a>

[![arXiv](https://img.shields.io/badge/arXiv-EAT-9065CA.svg?logo=arXiv)](http://arxiv.org/abs/2309.04946)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=data:image/svg%2bxml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBzdHJva2U9IiMwMDAwMDAiPgoNPGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiLz4KDTxnIGlkPSJTVkdSZXBvX3RyYWNlckNhcnJpZXIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgoNPGcgaWQ9IlNWR1JlcG9faWNvbkNhcnJpZXIiPiA8cGF0aCBkPSJNMyA2QzMgNC4zNDMxNSA0LjM0MzE1IDMgNiAzSDE0QzE1LjY1NjkgMyAxNyA0LjM0MzE1IDE3IDZWMTRDMTcgMTUuNjU2OSAxNS42NTY5IDE3IDE0IDE3SDZDNC4zNDMxNSAxNyAzIDE1LjY1NjkgMyAxNFY2WiIgc3Ryb2tlPSIjODFhOWQwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPiA8cGF0aCBkPSJNMjEgN1YxOEMyMSAxOS42NTY5IDE5LjY1NjkgMjEgMTggMjFINyIgc3Ryb2tlPSIjODFhOWQwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPiA8cGF0aCBkPSJNOSAxMlY4TDEyLjE0MjkgMTBMOSAxMloiIGZpbGw9IiM4MWE5ZDAiIHN0cm9rZT0iIzgxYTlkMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4gPC9nPgoNPC9zdmc+)](https://yuangan.github.io/eat/)
**<a href="https://colab.research.google.com/drive/133hwDHzsfRYl-nQCUQxJGjcXa5Fae22Z#scrollTo=GWqHlw6kKrbo"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="20" alt="google colab logo"></a>**
[![License](https://img.shields.io/badge/license-CC--BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub Stars](https://img.shields.io/github/stars/yuangan/EAT_code?style=social)](https://github.com/yuangan/EAT_code)

![EAT](./doc/web_intro2.gif)

</div>
<div align="justify">





**News:**
* 10/03/2024 Released all evaluation codes used in our paper, please refer to [here](https://github.com/yuangan/evaluation_eat) for more details.
* 26/12/2023 Released the **A2KP Training** code. Thank you for your attention and patience~ :tada:
* 05/12/2023 Released the LRW test code.
* 27/10/2023 Released the **Emotional Adaptation Training** code. Thank you for your patience~ :tada:
* 17/10/2023 Released the evaluation code for the MEAD test results. For more information, please refer to [evaluation_eat](https://github.com/yuangan/evaluation_eat).
* 21/09/2023 Released the preprocessing code. Now, EAT can generate emotional talking-head videos with <strong>any</strong> portrait and driven video.
* 07/09/2023 Released the pre-trained weight and inference code.
  
# Environment
If you wish to run only our demo, we recommend trying it out in [Colab](https://colab.research.google.com/drive/133hwDHzsfRYl-nQCUQxJGjcXa5Fae22Z#scrollTo=GWqHlw6kKrbo). Please note that our preprocessing and training code should be executed locally, and requires the following environmental configuration:

```conda/mamba env create -f environment.yml```

**Note**: We recommend using [mamba](https://github.com/conda-forge/miniforge#mambaforge) to install dependencies, which is faster than conda.

# Checkpoints && Demo dependencies
In the EAT_code folder, Use gdown or download and unzip the [ckpt](https://drive.google.com/file/d/1KK15n2fOdfLECWN5wvX54mVyDt18IZCo/view?usp=drive_link), [demo](https://drive.google.com/file/d/1MeFGC7ig-vgpDLdhh2vpTIiElrhzZmgT/view?usp=drive_link) and [Utils](https://drive.google.com/file/d/1HGVzckXh-vYGZEUUKMntY1muIbkbnRcd/view?usp=drive_link) to the specific folder.
```
gdown --id 1KK15n2fOdfLECWN5wvX54mVyDt18IZCo && unzip -q ckpt.zip -d ckpt
gdown --id 1MeFGC7ig-vgpDLdhh2vpTIiElrhzZmgT && unzip -q demo.zip -d demo
gdown --id 1HGVzckXh-vYGZEUUKMntY1muIbkbnRcd && unzip -q Utils.zip -d Utils
```

# Demo
Execute the code within our <strong>eat</strong> environment using the command: 

```conda activate eat```

Then, run the demo with:

```CUDA_VISIBLE_DEVICES=0 python demo.py --root_wav ./demo/video_processed/W015_neu_1_002 --emo hap```

- **Parameters**:
  - **root_wav**: Choose from ['obama', 'M003_neu_1_001', 'W015_neu_1_002', 'W009_sad_3_003', 'M030_ang_3_004']. Preprocessed wavs are located in ```./demo/video_processed/```. The 'obama' wav is approximately 5 minutes long, while the others are much shorter.
  - **emo**: Choose from ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

**Note 1**: Place your own images in ```./demo/imgs/``` and run the above command to generate talking-head videos with aligned new portraits. If you prefer not to align your portrait, simply place your cropped image (256x256) in ```./demo/imgs_cropped```. Due to the background used in the MEAD training set, results tend to be better with a similar background.

**Note 2**: To test with a custom audio, you need to replace the ```video_name/video_name.wav``` and deepspeech feature ```video_name/deepfeature32/video_name.npy```. The output length will depend on the shortest length of the audio and driven poses. Refer to [here](https://github.com/yuangan/EAT_code/blob/main/demo.py#L139) for more details.

**Note 3**: The audio used in our work should be sampled at 16,000 Hz and the corresponding video should have a frame rate of 25 fps.

# Test MEAD
To reproduce the results of MEAD as reported in our paper, follow these steps:

First, Download the additional MEAD test data from [mead_data](https://drive.google.com/file/d/1_6OfvP1B5zApXq7AIQm68PZu1kNyMwUY/view?usp=drive_link) and unzip it into the mead_data directory:
   
```gdown --id 1_6OfvP1B5zApXq7AIQm68PZu1kNyMwUY && unzip -q mead_data.zip -d mead_data```

Then, Execute the test using the following command:
   
```CUDA_VISIBLE_DEVICES=0 python test_mead.py [--part 0/1/2/3] [--mode 0]```

- **Parameters**:
  - **part**: Choose from [0, 1, 2, 3]. These represent the four test parts in the MEAD test data.
  - **mode**: Choose from [0, 1]. Where `0` tests only 100 samples in total, and `1` tests all samples (985 in total).

You can use our [evaluation_eat](test_posedeep_deepprompt_eam3d.sh) code to evaluate.

# Test LRW
To reproduce the results of LRW as reported in our paper, you need to download and extract the LRW **test** dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). Due to the limitations of the license, we cannot provide any video data. (The name of the test files can be found [here](https://drive.google.com/file/d/1WYXBLlp9jwnGsf0V8eQf3aKCXJLGeTS3/view?usp=sharing) for validation.) After downloading LRW, You will need to preprocess them using our [preprocessing code](https://github.com/yuangan/EAT_code/tree/main#preprocessing). Then, move and rename the output files as follows:

'imgs, latents, deepfeature32, poseimg, video_fps25/*.wavs' --> 'lrw/lrw_images, lrw/lrw_latent, lrw/lrw_df32, lrw/poseimg, lrw/lrw_wavs/*.wav'

Change the dataset path in [test_lrw_posedeep_normalize_neutral.py](https://github.com/yuangan/EAT_code/blob/main/test_lrw_posedeep_normalize_neutral.py#L45).

Then, execute the following command:

```CUDA_VISIBLE_DEVICES=0 python test_lrw_posedeep_normalize_neutral.py --name deepprompt_eam3d_all_final_313 --part [0/1/2/3] --mode 0```

or run them concurrently:

```bash test_lrw_posedeep_normalize_neutral.sh```

The results will be saved in './result_lrw/'.

# Preprocessing
If you want to test with your own driven video that includes audio, place your video (which should have audio) in the ```preprocess/video```. Then execute the preprocessing code:

```
cd preprocess
python preprocess_video.py
```

The video will be processed and saved in the ```demo/video_processed```. To test it, run:

```CUDA_VISIBLE_DEVICES=0 python demo.py --root_wav ./demo/video_processed/[fill in your video name] --emo [fill in emotion name]```

The videos should contain only one person. We will crop the input video according to the estimated landmark of the first frame. Refer to these [video](https://drive.google.com/file/d/1sAoplzY4b6JCW0JQHf_HKEL5luuWuGAk/view?usp=drive_link) for more details.

**Note 1**: The preprocessing code has been verified to work correctly with TensorFlow version 1.15.0, which can be installed on Python 3.7. Refer to this [issue]((https://github.com/YudongGuo/AD-NeRF/issues/69)) for more information.

**Note 2**: Extract the bbox for training with ```preprocess/extract_bbox.py```.

# A2KP Training
**Data&Ckpt Preparation:**
- Download Voxceleb 2 datasets from [the official website](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) or [here](https://zhuanlan.zhihu.com/p/588668145) and preprocess with our code. It needs to be noted that we exclude some videos, which have blurred or small faces, according to the face detected in the first frame.
- Modify the dataset path in [config/pretrain_a2kp_s1.yaml](https://github.com/yuangan/EAT_code/blob/main/config/pretrain_a2kp_s1.yaml#L2), [config/pretrain_a2kp_img_s2.yaml](https://github.com/yuangan/EAT_code/blob/main/config/pretrain_a2kp_img_s2.yaml#L2) and [frames_dataset_transformer25.py](https://github.com/yuangan/EAT_code/blob/main/frames_dataset_transformer25.py#L32C1-L35C33).
- Download ckpt [000299_1024-checkpoint.pth.tar](https://drive.google.com/file/d/1dBoUjensiX4Ic2jmvL6_nrUeGG7hdOP1/view?usp=sharing) to folder 'ckpt/'
- Download the [voxselect](https://drive.google.com/file/d/1ME7bcJIvFemBGpI6jdQq99PE0g7SnhIz/view?usp=drive_link) file and untar it to the processed ```vox_path```.

**Execution:**
- Run the following command to start training A2KP transformer with latent and pca loss in 4 GPUs:

  ```python pretrain_a2kp.py --config config/pretrain_a2kp_s1.yaml --device_ids 0,1,2,3 --checkpoint ./ckpt/pretrain_new_274.pth.tar```

- **Note:** Stop training when the loss converges. We trained for 8 epochs here. Our training log is at: ```./output/qvt_2 30_10_22_14.59.29/log.txt```. Copy and rename the output ckpt to folder ```./ckpt```, for example: ```ckpt/qvt_2_1030_281.pth.tar```
- Run the following command to start training A2KP transformer with all loss in 4 GPUs:
  
  ```python pretrain_a2kp_img.py --config config/pretrain_a2kp_img_s2.yaml --device_ids 0,1,2,3 --checkpoint ./ckpt/qvt_2_1030_281.pth.tar```

- **Note:** Stop training when the loss converges. We trained for 24 epochs here. Our training log is at: ```./output/qvt_img_pca_sync_4 01_11_22_15.47.54/log.txt```

# Emotional Adaptation Training

**Data&Ckpt Preparation:**
- The processed MEAD data used in our paper can be downloaded from [Yandex](https://disk.yandex.com/d/yzk1uTlZgwortw) or [Baidu](https://pan.baidu.com/s/1Jxzow2anGjMa-y3F8yQwAw?pwd=lsle#list/path=%2F). After downloading, concatenate, unzip the files, and update the paths in [deepprompt_eam3d_st_tanh_304_3090_all.yaml](https://github.com/yuangan/EAT_code/blob/main/config/deepprompt_eam3d_st_tanh_304_3090_all.yaml#L3) and [frames_dataset_transformer25.py](https://github.com/yuangan/EAT_code/blob/main/frames_dataset_transformer25.py#L32C1-L35C33).
- We have updated `environment.yaml` to adapt to the training environment. You can install the required packages using pip or mamba, or reinstall the `eat` environment.
- We have also updated `ckpt.zip`, which contains the pre-trained checkpoints that can be used directly for the second phase of training.

**Execution:**
- Run the following command to start training in 4 GPUs:

  ```python -u prompt_st_dp_eam3d.py --config ./config/deepprompt_eam3d_st_tanh_304_3090_all.yaml --device_ids 0,1,2,3 --checkpoint ./ckpt/qvt_img_pca_sync_4_01_11_304.pth.tar```

- **Note 1**: The `batch_size` in the config should be consistent with the number of GPUs. To compute the sync loss, we train consecutive `syncnet_T` frames (which is 5 in our paper) in a batch. Each GPU is assigned a batch during training, consuming around 17GB of VRAM.
- **Note 2**: Our checkpoints are saved every half an hour. The results in the paper were obtained using 4 Nvidia 3090 GPUs, training for about 5-6 hours. Please refer to `output/deepprompt_eam3d_st_tanh_304_3090_all\ 03_11_22_15.40.38/log.txt` for the training logs at that time. The convergence speed of the training loss should be similar to what is shown there.

**Evaluation:**
- The checkpoints and logs are saved at `./output/deepprompt_eam3d_st_tanh_304_3090_all [timestamp]`.
- Change the data root in [test_posedeep_deepprompt_eam3d.py](https://github.com/yuangan/EAT_code/blob/main/test_posedeep_deepprompt_eam3d.py#L41C1-L43C32) and dirname in [test_posedeep_deepprompt_eam3d.sh](https://github.com/yuangan/EAT_code/blob/main/test_posedeep_deepprompt_eam3d.sh#L1C1-L5C5), then run the following command for batch testing:
  
  ```bash test_posedeep_deepprompt_eam3d.sh```
  
- The results from sample testing (100 samples) are stored in `./result`. You can use our [evaluation_eat](test_posedeep_deepprompt_eam3d.sh) code to evaluate.

# Contact
Our code is under the CC-BY-NC 4.0 license and intended solely for research purposes. If you have any questions or wish to use it for commercial purposes, please contact us at ganyuan@zju.edu.cn and yangyics@zju.edu.cn

# Citation
If you find this code helpful for your research, please cite:
```
@InProceedings{Gan_2023_ICCV,
    author    = {Gan, Yuan and Yang, Zongxin and Yue, Xihang and Sun, Lingyun and Yang, Yi},
    title     = {Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22634-22645}
}
```

# Acknowledge
We acknowledge these works for their public code and selfless help: [EAMM](https://github.com/jixinya/EAMM), [OSFV (unofficial)](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [AVCT](https://github.com/FuxiVirtualHuman/AAAI22-one-shot-talking-face), [PC-AVS](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS), [Vid2Vid](https://github.com/NVIDIA/vid2vid), [AD-NeRF](https://github.com/YudongGuo/AD-NeRF) and so on.
</div>


