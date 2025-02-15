[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-as-queries-for-referring-video/referring-expression-segmentation-on-refer-1)](https://paperswithcode.com/sota/referring-expression-segmentation-on-refer-1?p=language-as-queries-for-referring-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-as-queries-for-referring-video/referring-expression-segmentation-on-a2d)](https://paperswithcode.com/sota/referring-expression-segmentation-on-a2d?p=language-as-queries-for-referring-video)

The official implementation of the **CVPR2022** paper: 

<div align="center">
<h1>
<b>
Language as Queries for Referring <br> Video Object Segmentation
</b>
</h1>
</div>

<p align="center"><img src="docs/network.png" width="800"/></p>

> [**Language as Queries for Referring Video Object Segmentation**](https://arxiv.org/abs/2201.00487)
>
> Jiannan Wu, Yi Jiang, Peize Sun, Zehuan Yuan, Ping Luo

### Abstract

In this work, we propose a simple and unified framework built upon Transformer, termed ReferFormer. It views the language as queries and directly attends to the most relevant regions in the video frames. Concretely, we introduce a small set of object queries conditioned on the language as the input to the Transformer. In this manner, all the queries are obligated to find the referred objects only. They are eventually transformed into dynamic kernels which capture the crucial object-level information, and play the role of convolution filters to generate the segmentation masks from feature maps. The object tracking is achieved naturally by linking the corresponding queries across frames. This mechanism greatly simplifies the pipeline and the end-to-end framework is significantly different from the previous methods. Extensive experiments on Ref-Youtube-VOS, Ref-DAVIS17, A2D-Sentences and JHMDB-Sentences show the effectiveness of ReferFormer. 

## Update
- **(2022/12/19)** We add the results on RefCOCO/+/g validation set.
- **(2022/07/31)** We upload the files for joint-training. 
- **(2022/04/04)** We upload the data conversion and main files for pre-training. 
- **(2022/03/11)** We upload the model on Ref-Youtube-VOS by jointly training Ref-Youtube-VOS and Ref-COCO/+/g, which leads to higher performance.
- **(2022/03/03)** ReferFormer is accepted by CVPR2022. 👏

## Demo

- Ref-DAVIS17

<img src="docs/davis_demo1.gif" width="400"/><img src="docs/davis_demo2.gif" width="400"/>

- Ref-Youtube-VOS

<img src="docs/ytvos_demo1.gif" width="400"/><img src="docs/ytvos_demo2.gif" width="400"/>



## Requirements

We test the codes in the following environments, other versions may also be compatible:

- CUDA 11.1
- Python 3.7
- Pytorch 1.8.1


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data.md](docs/data.md) for data preparation.

We provide the pretrained model for different visual backbones. You may download them [here](https://drive.google.com/drive/folders/19tV2TBtuji052Tq6eVPJvlec39wKqX9O?usp=sharing) and put them in the directory `pretrained_weights`.

<!-- For the Swin Transformer and Video Swin Transformer backbones, the weights are intialized using the pretrained model provided in the repo [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer). For your convenience, we upload the pretrained model in the google drives [swin_pretrained](https://drive.google.com/drive/u/0/folders/1QWLayukDJYAxTFk7NPwerfso3Lrx35NL) and [video_swin_pretrained](https://drive.google.com/drive/u/0/folders/19qb9VbKSjuwgxsiPI3uv06XzQkB5brYM). -->


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
├── pretrained_weights/
├── eval_davis.py
├── main.py
├── engine.py
├── inference_ytvos.py
├── inference_davis.py
├── opts.py
...
```

## Model Zoo

All the models are trained using 8 NVIDIA Tesla V100 GPU. You may change the `--backbone` parameter to use different backbones (see [here](https://github.com/wjn922/ReferFormer/blob/232b4066fb7d10845e4083e6a5a2cc0af5d1757e/opts.py#L31)).

**Note:** If you encounter the `OOM` error, please add the command `--use_checkpoint` (we add this command for Swin-L, Video-Swin-S and Video-Swin-B models).


### Ref-Youtube-VOS

To evaluate the results, please upload the zip file to the [competition server](https://competitions.codalab.org/competitions/29139#participate-submit_results).

| Backbone| J&F | CFBI J&F  | Pretrain | Model | Submission | CFBI Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 55.6 | 59.4 | [weight](https://drive.google.com/file/d/1zFBTmWhHQn_QxN84hpT-fA0C5TzZUAZK/view?usp=sharing) | [model](https://drive.google.com/file/d/1TBykzTRiOEbpM2YEYnDICyqjwkvlD2cR/view?usp=sharing) | [link](https://drive.google.com/file/d/10jlaCgIA9D91kb7d5_En-iUusoKU02mh/view?usp=sharing) | [link](https://drive.google.com/file/d/1H3m6jAXZNXIxEnIMpkvb_iYb62v2hAn2/view?usp=sharing) |
| ResNet-101 | 57.3 | 60.3 | [weight](https://drive.google.com/file/d/1FKIdXM9UEHSn93Xhbqi2hYBktDRMBsLr/view?usp=sharing) | [model](https://drive.google.com/file/d/1QJU8HC8xk_3gMcj-Yi4vkUHaeHeP8n82/view?usp=sharing) | [link](https://drive.google.com/file/d/15DZ0QXCdFmg0GHM0CgMAH20ZVbuCaBEW/view?usp=sharing) | [link](https://drive.google.com/file/d/1FCZcRqoPXdTnoODQxre7pGOIBCuWNv_v/view?usp=sharing) |
| Swin-T | 58.7 | 61.2 | [weight](https://drive.google.com/file/d/1TcGsseTVftkpdw_glFOwG-Zvg3NMEPuw/view?usp=sharing) | [model](https://drive.google.com/file/d/1AOHK6Qp8T0rUzgHT9twvWn4oGLS5MnuS/view?usp=sharing) | [link](https://drive.google.com/file/d/1kvzSLtDQTZjPLQ-9YKy8YmhN_JKheSmJ/view?usp=sharing) | [link](https://drive.google.com/file/d/1y3eegr7aB2R80L8WfRhcD-UfS3KF3U-5/view?usp=sharing) |
| Swin-L | 62.4 | 63.3 | [weight](https://drive.google.com/file/d/1pM7mBgvHtI_XNbfkX5eKT4jsUaLL3Qa4/view?usp=sharing) | [model](https://drive.google.com/file/d/103N96BbQnT9P57aGx7TCsZVyGr3ejFdl/view?usp=sharing) | [link](https://drive.google.com/file/d/1o18UzDUCy-NtoIGMxsD9VGu7y3H5HADk/view?usp=sharing) | [link](https://drive.google.com/file/d/1MnIibGGh9IhQ00qiqU7VCTH9CWE1ron3/view?usp=sharing) |
| Video-Swin-T* | 56.0 | - | - | [model](https://drive.google.com/file/d/14X_YysCXv13LzLoWy9ViIn623KQ-ol51/view?usp=sharing) | [link](https://drive.google.com/file/d/1TuADkEDibY8gW3On0K8fzi8-9JPnio0W/view?usp=sharing) | - |
| Video-Swin-T | 59.4 | - | [weight](https://drive.google.com/file/d/1qaEuOocLOZoj89unhZP3oN708ofmqopS/view?usp=sharing) | [model](https://drive.google.com/file/d/1vkG8nyUue9XOnScUMKo__khhxvuJP9R6/view?usp=sharing) | [link](https://drive.google.com/file/d/1LlZ00AhWzzW5uDyBy21qvZTJ2gnbC2r4/view?usp=sharing) | - |
| Video-Swin-S | 60.1 | - | [weight](https://drive.google.com/file/d/1SHzlzxE-KQeoGoYucdnPZtw_uz2ymgxn/view?usp=sharing) | [model](https://drive.google.com/file/d/1R0C5hYck6OwKko9tXOpIQ02SuE4mb1xj/view?usp=sharing) | [link](https://drive.google.com/file/d/1gcN_oD3-qh5wSuqUeDtwQpuwaWdJ9A6k/view?usp=sharing) | - |
| Video-Swin-B | 62.9 | - |[weight](https://drive.google.com/file/d/1gbyknvuOiKpxK1kljxFnt1k5Jxp3R1jv/view?usp=sharing)  | [model](https://drive.google.com/file/d/1_dS8hyZJFdijvsDVmoVcHjWhRE6sAXvS/view?usp=sharing) | [link](https://drive.google.com/file/d/1H8r3DgKeVuoh4egFeE9jS7ytknbD_xSb/view?usp=sharing) | - |

\* indicates the model is trained from scratch.

Joint training with Ref-COCO/+/g datasets.
| Backbone| J&F | J | F | Model | Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 58.7 | 57.4 | 60.1 | [model](https://drive.google.com/file/d/1GAzI9zLEQKxp5KQHu1FOuqlw5PwAhMey/view?usp=sharing) | [link](https://drive.google.com/file/d/1cELD6A7zs0POgIW5_xZIX-pCDdmmn5WP/view?usp=sharing) |
| ResNet-101 | 59.3 | 58.1 | 60.4 | [model](https://drive.google.com/file/d/1R4PThhYGqaanm3h46gygQ6strHVJBpn5/view?usp=sharing) | [link](https://drive.google.com/file/d/1S1uxYbM_dLddtZRKxfMdXZhKpEv1hkZ3/view?usp=sharing) |
| Swin-L | 64.2 | 62.3 | 66.2 | [model](https://drive.google.com/file/d/1_nGi5JjlRcOUI8kXXduOE5iv0apnSXgR/view?usp=sharing) | [link](https://drive.google.com/file/d/1SL1JdM0zutNkef7x_9lUr0hbn25dMMZV/view?usp=sharing) |
| Video-Swin-T | 62.6 | 59.9 | 63.3 | [model](https://drive.google.com/file/d/1gPVVdAuX_0-H2McPPJYX6HldGQtekD8b/view?usp=sharing) | [link](https://drive.google.com/file/d/1d-XVhMlTOZDQJC_4a9rKzan5rMdxO1yl/view?usp=sharing) |
| Video-Swin-S | 63.3 | 61.4 | 65.2 | [model](https://drive.google.com/file/d/1zm9URDfL19xi5LN9vCM2c58-W-A5RCz_/view?usp=sharing) | [link](https://drive.google.com/file/d/10Qz1jpNVrHaXYxXs4AVH_2Zqut2d0WAD/view?usp=sharing) |
| Video-Swin-B | 64.9 | 62.8 | 67.0 | [model](https://drive.google.com/file/d/183AJLTtzwL15r-hhG9hUFrK5tfq37BH1/view?usp=sharing) | [link](https://drive.google.com/file/d/1CKCgoTbg6PwVZRzQ80CMiieq9NEtPNWD/view?usp=sharing) |




### Ref-DAVIS17

As described in the paper, we report the results using the model trained on Ref-Youtube-VOS without finetune.

| Backbone| J&F | J | F | Model | 
| :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 58.5 | 55.8 | 61.3 | [model](https://drive.google.com/file/d/1TBykzTRiOEbpM2YEYnDICyqjwkvlD2cR/view?usp=sharing) |
| Swin-L | 60.5 | 57.6 | 63.4 | [model](https://drive.google.com/file/d/103N96BbQnT9P57aGx7TCsZVyGr3ejFdl/view?usp=sharing) |
| Video-Swin-B | 61.1 | 58.1 | 64.1 | [model](https://drive.google.com/file/d/1_dS8hyZJFdijvsDVmoVcHjWhRE6sAXvS/view?usp=sharing) |


### A2D-Sentences

The pretrained models are the same as those provided for Ref-Youtube-VOS.

| Backbone| Overall IoU | Mean IoU | mAP  | Pretrain | Model |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Video-Swin-T* | 72.3 | 64.1 | 48.6 | - | [model](https://drive.google.com/file/d/1lK3Hulc8K6croRTrzJmz8QMhndQYDzsf/view?usp=sharing) \| [log](https://drive.google.com/file/d/1_uzt6iI6TfM3cQbIblOI5_siFDxCa4mj/view?usp=sharing) |
| Video-Swin-T | 77.6 | 69.6 | 52.8 | [weight](https://drive.google.com/file/d/1qaEuOocLOZoj89unhZP3oN708ofmqopS/view?usp=sharing) | [model](https://drive.google.com/file/d/1b56oyvJZTY70ztCgDhD5vBG7fZohqzYE/view?usp=sharing) \| [log](https://drive.google.com/file/d/1n_Swnok9I2uaZtCHpBzAC53J3zILh0Rk/view?usp=sharing) |
| Video-Swin-S | 77.7 | 69.8 | 53.9 | [weight](https://drive.google.com/file/d/1SHzlzxE-KQeoGoYucdnPZtw_uz2ymgxn/view?usp=sharing) | [model](https://drive.google.com/file/d/1edp5ZjCr4OG-8E9Co2_UgknoKMWUCB5y/view?usp=sharing) \| [log](https://drive.google.com/file/d/1sOXhiCILSJTAiPYixyEQZLInzLfvFHXb/view?usp=sharing) |
| Video-Swin-B | 78.6 | 70.3 | 55.0 | [weight](https://drive.google.com/file/d/1gbyknvuOiKpxK1kljxFnt1k5Jxp3R1jv/view?usp=sharing) | [model](https://drive.google.com/file/d/11UdnixY1ZQLpFqEq3_mdM4ELXGhrG7YD/view?usp=sharing) \| [log](https://drive.google.com/file/d/1UcSGzKKDHoClHv87tg4sQi4sbH4Vr01j/view?usp=sharing) |

\* the model is trained from scratch and set `--num_frames 6`.


### JHMDB-Sentences

As described in the paper, we report the results using the model trained on A2D-Sentences without finetune.

| Backbone| Overall IoU | Mean IoU | mAP  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| Video-Swin-T* | 70.0 | 69.3 | 39.1 | [model](https://drive.google.com/file/d/1lK3Hulc8K6croRTrzJmz8QMhndQYDzsf/view?usp=sharing) | 
| Video-Swin-T | 71.9 | 71.0 | 42.2 | [model](https://drive.google.com/file/d/1b56oyvJZTY70ztCgDhD5vBG7fZohqzYE/view?usp=sharing) |
| Video-Swin-S | 72.8 | 71.5 | 42.4 | [model](https://drive.google.com/file/d/1edp5ZjCr4OG-8E9Co2_UgknoKMWUCB5y/view?usp=sharing) |
| Video-Swin-B | 73.0 | 71.8 | 43.7 | [model](https://drive.google.com/file/d/11UdnixY1ZQLpFqEq3_mdM4ELXGhrG7YD/view?usp=sharing) | 

\* the model is trained from scratch and set `--num_frames 6`.


### RefCOCO/+/g

We also support evaluate on RefCOCO/+/g validation set by using the pretrained weights (num_frames=1).
Specifically, we measure the P@0.5 and overall IoU (oIoU) for REC and RIS tasks, respectively.

REC (referring epression understanding):

| Backbone| RefCOCO | RefCOCO+ | RefCOCOg  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 85.0 | 79.2 | 79.0 | [weight](https://drive.google.com/file/d/1zFBTmWhHQn_QxN84hpT-fA0C5TzZUAZK/view?usp=sharing) | 
| ResNet-101 | 85.4 | 75.8 | 79.9 | [weight](https://drive.google.com/file/d/1FKIdXM9UEHSn93Xhbqi2hYBktDRMBsLr/view?usp=sharing) |
| Swin-T | 86.7 | 77.2 | 80.6 | [weight](https://drive.google.com/file/d/1TcGsseTVftkpdw_glFOwG-Zvg3NMEPuw/view?usp=sharing) |
| Swin-L | 89.8 | 80.0 | 83.9 | [weight](https://drive.google.com/file/d/1pM7mBgvHtI_XNbfkX5eKT4jsUaLL3Qa4/view?usp=sharing) | 

RIS (referring image segmentation):

| Backbone| RefCOCO | RefCOCO+ | RefCOCOg  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 71.1 | 64.1 | 64.1 | [weight](https://drive.google.com/file/d/1zFBTmWhHQn_QxN84hpT-fA0C5TzZUAZK/view?usp=sharing) | 
| ResNet-101 | 71.8 | 61.1 | 64.9 | [weight](https://drive.google.com/file/d/1FKIdXM9UEHSn93Xhbqi2hYBktDRMBsLr/view?usp=sharing) |
| Swin-T | 72.9 | 62.4 | 66.1 | [weight](https://drive.google.com/file/d/1TcGsseTVftkpdw_glFOwG-Zvg3NMEPuw/view?usp=sharing) |
| Swin-L | 77.1 | 65.8 | 69.3 | [weight](https://drive.google.com/file/d/1pM7mBgvHtI_XNbfkX5eKT4jsUaLL3Qa4/view?usp=sharing) | 


## Get Started

Please see [Ref-Youtube-VOS](docs/Ref-Youtube-VOS.md), [Ref-DAVIS17](docs/Ref-DAVIS17.md), [A2D-Sentences](docs/A2D-Sentences.md) and [JHMDB-Sentences](docs/JHMDB-Sentences.md) for details.



## Acknowledgement

This repo is based on [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [VisTR](https://github.com/Epiphqny/VisTR). We also refer to the repositories [MDETR](https://github.com/ashkamath/mdetr) and [MTTR](https://github.com/mttr2021/MTTR). Thanks for their wonderful works.


## Citation

```
@article{wu2022referformer,
      title={Language as Queries for Referring Video Object Segmentation}, 
      author={Jiannan Wu and Yi Jiang and Peize Sun and Zehuan Yuan and Ping Luo},
      journal={arXiv preprint arXiv:2201.00487},
      year={2022},
}
```

