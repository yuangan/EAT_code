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
* 07/09/2023 Release the pre-trained weight and inference code.
* 21/09/2023 Release the preprocessing code. Now, EAT can generate emotional talking-head videos with <strong>any</strong> portrait and driven video. Thank you for your patience~:tada:

# Environment
For the quickest configuration, it's recommended to try the demo in [Colab](https://colab.research.google.com/drive/133hwDHzsfRYl-nQCUQxJGjcXa5Fae22Z#scrollTo=GWqHlw6kKrbo).

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

**Note 2**: Replace the ```video_name/video_name.wav``` and deepspeech feature ```video_name/deepfeature32/video_name.npy```, you can test with a new wav. The output length will depend on the shortest length of the audio and driven poses. Refer to [here](https://github.com/yuangan/EAT_code/blob/main/demo.py#L139) for more details.

# Test MEAD
To reproduce the results of MEAD as reported in our paper, follow these steps:

First, Download the additional MEAD test data from [mead_data](https://drive.google.com/file/d/1_6OfvP1B5zApXq7AIQm68PZu1kNyMwUY/view?usp=drive_link) and unzip it into the mead_data directory:
   
```gdown --id 1_6OfvP1B5zApXq7AIQm68PZu1kNyMwUY && unzip -q mead_data.zip -d mead_data```

Then, Execute the test using the following command:
   
```CUDA_VISIBLE_DEVICES=0 python test_mead.py [--part 0/1/2/3] [--mode 0]```

- **Parameters**:
  - **part**: Choose from [0, 1, 2, 3]. These represent the four test parts in the MEAD test data.
  - **mode**: Choose from [0, 1]. Where `0` tests only 100 samples in total, and `1` tests all samples (985 in total).

# Preprocessing
If you want to test with your own driven video that includes audio, place your video (which should have audio) in the ```preprocess/video```. Then execute the preprocessing code:

```
cd preprocess
python preprocess_video.py
```

The video will be processed and saved in the ```demo/video_processed```. To test it, run:

```CUDA_VISIBLE_DEVICES=0 python demo.py --root_wav ./demo/video_processed/[fill in your video name] --emo [fill in emotion name]```

The videos should contain only one person. We will crop the input video according to the estimated landmark of the first frame. Refer to these [video](https://drive.google.com/file/d/1sAoplzY4b6JCW0JQHf_HKEL5luuWuGAk/view?usp=drive_link) for more details.

**Note**: The preprocessing code has been verified to work correctly with TensorFlow version 1.15.0, which can be installed on Python 3.7. Refer to this [issue]((https://github.com/YudongGuo/AD-NeRF/issues/69)) for more information. We have updated the environment.yml file for preprocessing; please reinstall the **eat** environment if needed.

**TODO:**
* Evaluation Code
* Training Dataset
* Baselines
* Training Code

# Contact
Our code is intended solely for research purposes. If you have any questions or wish to use it for commercial purposes, please contact me at ganyuan@zju.edu.cn

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


