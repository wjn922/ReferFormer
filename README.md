[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

The official implementation of the paper: 

<div align="center">
<h1>
<b>
Language as Queries for Referring Object Segmentation
</b>
</h1>
</div>

<p align="center"><img src="docs/network.png" width="500"/></p>

> [**Language as Queries for Referring Video Object Segmentation**](todo)
>
> Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, Ping Luo

### Abstract

In this work, we propose a simple and unified framework built upon Transformer, termed ReferFormer. It views the language as queries and directly attends to the most relevant regions in the video frames. Concretely, we introduce a small set of object queries conditioned on the language as the input to the Transformer. In this manner, all the queries are obligated to find the referred objects only. They are eventually transformed into dynamic kernels which capture the crucial object-level information, and play the role of convolution filters to generate the segmentation masks from feature maps. The object tracking is achieved naturally by linking the corresponding queries across frames. This mechanism greatly simplifies the pipeline and the end-to-end framework is significantly different from the previous methods. Extensive experiments on Ref-Youtube-VOS, Ref-DAVIS17, A2D-Sentences and JHMDB-Sentences show the effectiveness of ReferFormer. 

## Requirements

We test the code in the following environments, other versions may also be compatible:

- CUDA 11.1
- Python 3.7
- Pytorch 1.8.1


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data.md](docs/data.md) for data preparation.

For the Swin Transformer and Video Swin Transformer backbones, the weights are intialized using the pretrained model provided in the repo [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). For your convenience, we upload the pretrained model in the google drives [swin_pretrained](https://drive.google.com/drive/u/0/folders/1QWLayukDJYAxTFk7NPwerfso3Lrx35NL) and [video_swin_pretrained](https://drive.google.com/drive/u/0/folders/19qb9VbKSjuwgxsiPI3uv06XzQkB5brYM).


After the organization, we expect the directory struture to be the following:

```
ReferFormer/
├── data/
│   ├── ref-youtube-vos/
│   ├── ref-davis/
│   ├── a2d_sentences/
│   ├── jhmdb_sentences/
├── davis2017/
├── datasets/
├── models/
├── scipts/
├── tools/
├── util/
├── swin_pretrained/
├── video_swin_pretrained/
├── eval_davis.py
├── main.py
├── engine.py
├── inference_ytvos.py
├── inference_davis.py
├── opts.py
...
```

## Model Zoo

All the models are trained using 8 NVIDIA Tesla V100 GPU. You may change the `args` parameters to use different backbones. 
**Note:** If you encounter the `OOM` error, please add the command `--use_checkpoint` (For the Swin-L, Video-Swin-S and Video-Swin-B models, we use this command).


### Ref-Youtube-VOS



### Ref-DAVIS17

| Backbone| J&F | CFBI J&F  | Pretrain | Model | Submission | CFBI Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 55.6 | 59.4 |  |  |  | |
| ResNet-101 | 57.3 | 60.3 |  |  |  | |
| Swin-T | 58.7 | 61.2 |  |  |  | |
| Swin-L | 62.4 | 63.3 |  |  |  | |
| Video-Swin-T* | 55.8 |  |  |  |  | |
| Video-Swin-T | 59.4 |  |  |  |  | |
| Video-Swin-S | 60.1 |  |  |  |  | |
| Video-Swin-B | 62.9 |  |  |  |  | |

### A2D-Sentences

### JHMDB-Sentences


## Get Started

## Acknowledgement

This repo is based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [VisTR](https://github.com/Epiphqny/VisTR). We also refer to the repositories [MDETR](https://github.com/ashkamath/mdetr) and [MTTR](https://github.com/mttr2021/MTTR). Thanks for their wonderful works.


## Citation
