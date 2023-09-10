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

[![arXiv](https://img.shields.io/badge/arXiv-EAT-9065CA.svg?logo=arXiv)]()
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=data:image/svg%2bxml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiBzdHJva2U9IiMwMDAwMDAiPgoNPGcgaWQ9IlNWR1JlcG9fYmdDYXJyaWVyIiBzdHJva2Utd2lkdGg9IjAiLz4KDTxnIGlkPSJTVkdSZXBvX3RyYWNlckNhcnJpZXIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgoNPGcgaWQ9IlNWR1JlcG9faWNvbkNhcnJpZXIiPiA8cGF0aCBkPSJNMyA2QzMgNC4zNDMxNSA0LjM0MzE1IDMgNiAzSDE0QzE1LjY1NjkgMyAxNyA0LjM0MzE1IDE3IDZWMTRDMTcgMTUuNjU2OSAxNS42NTY5IDE3IDE0IDE3SDZDNC4zNDMxNSAxNyAzIDE1LjY1NjkgMyAxNFY2WiIgc3Ryb2tlPSIjODFhOWQwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPiA8cGF0aCBkPSJNMjEgN1YxOEMyMSAxOS42NTY5IDE5LjY1NjkgMjEgMTggMjFINyIgc3Ryb2tlPSIjODFhOWQwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPiA8cGF0aCBkPSJNOSAxMlY4TDEyLjE0MjkgMTBMOSAxMloiIGZpbGw9IiM4MWE5ZDAiIHN0cm9rZT0iIzgxYTlkMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4gPC9nPgoNPC9zdmc+)](https://yuangan.github.io/eat/)
**<a href="https://colab.research.google.com/drive/133hwDHzsfRYl-nQCUQxJGjcXa5Fae22Z#scrollTo=GWqHlw6kKrbo"><img src="https://colab.research.google.com/assets/colab-badge.svg" height="20" alt="google colab logo"></a>**
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![GitHub Stars](https://img.shields.io/github/stars/yuangan/EAT_code?style=social)](https://github.com/yuangan/EAT_code)

![EAT](./doc/web_intro2.gif)

</div>
<div align="justify">





**News:**
* 07/09/2023 Release the pre-trained weight and inference code.

# Environment
Recommend trying the demo in [Colab](https://colab.research.google.com/drive/133hwDHzsfRYl-nQCUQxJGjcXa5Fae22Z#scrollTo=GWqHlw6kKrbo) for the quickest configuration.

Recommend to use [mamba](https://github.com/conda-forge/miniforge#mambaforge), faster than conda, to install dependencies. 

```conda/mamba env create -f environment.yml```

# Checkpoints && Demo dependencies
In the EAT_code folder, Use gdown or download and unzip the [ckpt](https://drive.google.com/file/d/1KK15n2fOdfLECWN5wvX54mVyDt18IZCo/view?usp=drive_link), [demo](https://drive.google.com/file/d/1MeFGC7ig-vgpDLdhh2vpTIiElrhzZmgT/view?usp=drive_link) and [Utils](https://drive.google.com/file/d/1HGVzckXh-vYGZEUUKMntY1muIbkbnRcd/view?usp=drive_link) to the specific folder.
```
gdown --id 1KK15n2fOdfLECWN5wvX54mVyDt18IZCo && unzip -q ckpt.zip -d ckpt
gdown --id 1MeFGC7ig-vgpDLdhh2vpTIiElrhzZmgT && unzip -q demo.zip -d demo
gdown --id 1HGVzckXh-vYGZEUUKMntY1muIbkbnRcd && unzip -q Utils.zip -d Utils
```

# Run demo
Run the code under our <strong>eat</strong> environment with ```conda activate eat```.

```CUDA_VISIBLE_DEVICES=0 python demo.py --root_wav ./demo/video_processed/W015_neu_1_002 --emo hap```

<strong>root_wav</strong>: ['obama', 'M003_neu_1_001', 'W015_neu_1_002', 'W009_sad_3_003', 'M030_ang_3_004'] (preprocessed wavs are at ./demo/video_processed/. The obama wav is about 5 mins, while others are much shorter.)

<strong>emo</strong>: ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

If you want to process your video, please let us know. We will publish the pre-process code as soon as possible.

**TODO:**
* Add MEAD test
* Preprocess Code
* Evaluation Code
* Training Dataset
* Baselines

# Acknowledge
We acknowledge these works for their public code and selfless help: [EAMM](https://github.com/jixinya/EAMM), [OSFV(unofficial)](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [AVCT](https://github.com/FuxiVirtualHuman/AAAI22-one-shot-talking-face), [PC-AVS](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS) and so on.
</div>


