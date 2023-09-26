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

We provide the pretrained model for different visual backbones. You may download them [here](https://drive.google.com/drive/u/0/folders/11_qps3q75aH41IYHlXToyeIBUKkfdqso) and put them in the directory `pretrained_weights`.

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
| ResNet-50 | 55.6 | 59.4 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EepGibYBfyRGt_QedfE9SywBLF3v-bjoxo2R9E9YDqmIcw?e=7J7k1J) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EVRsV76e78lKuekbMLHgwlsBdG09pRVafEuBPN_wKXjJ1Q?e=SMeZlS) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EZ8tt46rv4xIjoiUkHGGPjwB1Yi6w2H-9BBVTyINOINmgQ?e=yWbDjp) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EZp0dd70UCNGvla2g25lTawB2AZyCDPN7QMl_KeESI5dkQ?e=1BfD2W) |
| ResNet-101 | 57.3 | 60.3 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESTAK4QCkMdNkVlQz1dd7GoBo3n_i9K4_FK4YLFBAFvBrg?e=Y3PlD5) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EaHNEx5MWR9HjTNh__W3IlYBIfhGd-nHKrshJ-MOyvofdw?e=shM4Ok) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EbrNhmt-wiNIv2tmQ-gOupgBrSBzhM1OJlNvid0J_8cPJg?e=8Fgets) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EWSPiUjgmORMuyaL91ueY1oBl159pO4k7RQYF-9eWrSJ-A?e=81hzDF) |
| Swin-T | 58.7 | 61.2 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESdasB6JLydDrs6mf68FrLMBuQBLBF7y_uxdveWl9oK68w?e=H5zeqk) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EUxJmp6QYR5LoUK12Wj55E0Bm0o6_9zl3OvOBN5KE9kJkg?e=SRS0qL) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EUMveO7cX1VAq48IAk9c6zoBc_Zy5f1kwa5h6C9q4LYt0A?e=iz9uMg) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EcnHrx4S5KVPqFYhr9CCARoBftAxdtldaWyGQAougBFnig?e=KG1LDq) |
| Swin-L | 62.4 | 63.3 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESngRLeZfV1LtrlZ7x5cVo4BR5_deWfov4Igt28LZGoDew?e=AVAsws) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EcCfv66Vl0xDl-rFukByXyQBEFNRTyLeVEKoeWrIvXmjNg?e=GcVTIr) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EdI15ujU4UpBilI4wt5lUQQB98JOq6KnMV5GHh77QiAn-w?e=o91ITz) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ETmJUpRGgyFHlGdEhcXqzekBDAfbFTExfHtmA4wHKCOkLw?e=l951Ea) |
| Video-Swin-T* | 56.0 | - | - | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EYXL3SKYOsRGtfSN-Wr9JCUBDvcXbbp67Sa4hs5dEDplxw?e=g2hGWo) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EUosvwAGikhGsyTPEOELMjEBQM-HZOaJ3fqcJjG2SV-5YA?e=vSUD12) | - |
| Video-Swin-T | 59.4 | - | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EUNTvEGXlsdLv3gicAbHfN0Ba23kcyy2-Z15IJTDLXKx_A?e=GqAYxT) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EZI2zogC5mtDu3KL5MVIaXIBzG3_3yTthoqyxjfTsGrvzA?e=lT5sVp) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EVKtr-5ZK5NIhhTvaUXGdRcBcHEGahAevUh1YCO2nvFfaQ?e=9Am7dc) | - |
| Video-Swin-S | 60.1 | - | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/Eb015DXX1LhDpiDoojxJTu8BBQ8ACicpVS8gwFStRJDK1w?e=NC368q) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EZI2zogC5mtDu3KL5MVIaXIBzG3_3yTthoqyxjfTsGrvzA?e=QEAdwh) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EUUJn8Zu7mlCnxLP8eNSbpIBvoEqz88EOg3y9ftQHhAhCw?e=RnSwxX) | - |
| Video-Swin-B | 62.9 | - |[weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ETDj4aGm_pRMuz8hLBi9Jy0BEFnsco0Uoz5qQEhWrxdNKQ?e=kKImMX)  | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EduJ_zS-Vd5Hn1qexxv5_mYBKX_8kRBOeX6dlfhED_GSwg?e=TxTWHb) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EZKduAM1fLpJrLK7l762xZ8BesK7zWKBjR0b9dFbCWhbfQ?e=SlAdyg) | - |

\* indicates the model is trained from scratch.

Joint training with Ref-COCO/+/g datasets.
| Backbone| J&F | J | F | Model | Submission | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 58.7 | 57.4 | 60.1 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EcxDd8USU4BGo_HlgukKiG4BXLvetkjLdi3_-N-3SpjMvw?e=tAPNFv) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EYmORJYVsUJLp8NnbtfnZigBCM-IJ5oomZZrXEbNPhIyww?e=Bh5eYx) |
| ResNet-101 | 59.3 | 58.1 | 60.4 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EShgDd650nBBsfoNEiUbybcB84Ma5NydxOucISeCrZmzHw?e=YOSszd) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EcW6Lt67k0RCjr_FT2XOxVcBUcrFSlFJo19-YdFZpBxOsg?e=avszXt) |
| Swin-L | 64.2 | 62.3 | 66.2 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/Ec_qxgvukuBPr-GQ_3gNcX0B8VCHCqIUvXX-0ydtk1s7HQ?e=7X99M1) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EbNV0kBQ7ZVDrfRafG6B3CwBbpM-yMJtQ9jI01HwEgWXBQ?e=FzoSrT) |
| Video-Swin-T | 62.6 | 59.9 | 63.3 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EdCVQzM4HxxIvdZUBLiNpBwBrcPTLlFEqxHVxOzx0geF3A?e=1ZSZvK) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EdAT37_CDDZKkbC1U9MDxTYBkR1DVwTn0zxzqEvgrG-5ig?e=6P065H) |
| Video-Swin-S | 63.3 | 61.4 | 65.2 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EdYbp2xp-xFFuolQopvILNMBYRq88ksNjpcv-zKfGzHxbA?e=NqRzTf) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EU6d1rGtkfBFkIoA-xUH2koBwdKW2fPCghYTzzd49KvFLQ?e=FMsJLT) |
| Video-Swin-B | 64.9 | 62.8 | 67.0 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EY3-adM5LptFj--klo5gWgsBhpSDOps91j-C81sBI8i9Hw?e=n19q0w) | [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EcSdF-jsBmZLn7iUzc3zXTUBnlfnXDFxPP7mtRbC1ttJwg?e=0wzR0t) |




### Ref-DAVIS17

As described in the paper, we report the results using the model trained on Ref-Youtube-VOS without finetune.

| Backbone| J&F | J | F | Model | 
| :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 58.5 | 55.8 | 61.3 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EVRsV76e78lKuekbMLHgwlsBdG09pRVafEuBPN_wKXjJ1Q?e=SMeZlS) |
| Swin-L | 60.5 | 57.6 | 63.4 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EcCfv66Vl0xDl-rFukByXyQBEFNRTyLeVEKoeWrIvXmjNg?e=GcVTIr) |
| Video-Swin-B | 61.1 | 58.1 | 64.1 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EduJ_zS-Vd5Hn1qexxv5_mYBKX_8kRBOeX6dlfhED_GSwg?e=TxTWHb) |


### A2D-Sentences

The pretrained models are the same as those provided for Ref-Youtube-VOS.

| Backbone| Overall IoU | Mean IoU | mAP  | Pretrain | Model |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Video-Swin-T* | 72.3 | 64.1 | 48.6 | - | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EfJs5WPRKfxEvifnIO3impABNgydbiO5qqI_uCF6LYKlCQ?e=mSRLCQ) \| [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/wjn922_connect_hku_hk/EVJyHq6zy6ZGuxE--K9nECwB333gFkP9vjXKjh9Mt0otcA?e=Kwnngd) |
| Video-Swin-T | 77.6 | 69.6 | 52.8 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EUNTvEGXlsdLv3gicAbHfN0Ba23kcyy2-Z15IJTDLXKx_A?e=GqAYxT) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/Ed3po2mJGQZHivGwMJJg8oMBumXm3Ye94oPH6wfRFK1d8A?e=NG2E9c) \| [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/wjn922_connect_hku_hk/EfO50qMduZNGvFcYJdRVKzABIJ8ZHhMiKWWvmDM14K9mnw?e=dgInSK) |
| Video-Swin-S | 77.7 | 69.8 | 53.9 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/Eb015DXX1LhDpiDoojxJTu8BBQ8ACicpVS8gwFStRJDK1w?e=NC368q) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EbAiydTvu41KsMYBEFzy_d8B0Nyy1fIf2tWG7Ao-FYD0Ug?e=tmaVAu) \| [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/wjn922_connect_hku_hk/EZl6sHhFDTBMgVGKVp18sqwBouTTnwPdirWId4PR6klTfg?e=17lDVV) |
| Video-Swin-B | 78.6 | 70.3 | 55.0 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ETDj4aGm_pRMuz8hLBi9Jy0BEFnsco0Uoz5qQEhWrxdNKQ?e=kKImMX) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EeP1aneDDbBCo9HnMTbNjsgBpMqrgfIzJzF_jVROpZ2GWQ?e=YmkNHC) \| [log](https://connecthkuhk-my.sharepoint.com/:t:/g/personal/wjn922_connect_hku_hk/EUnV-O_IAe5Mkyupsd7NosMBxUg8OjqepmQbpbV0PFB4gQ?e=W14suT) |

\* the model is trained from scratch and set `--num_frames 6`.


### JHMDB-Sentences

As described in the paper, we report the results using the model trained on A2D-Sentences without finetune.

| Backbone| Overall IoU | Mean IoU | mAP  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| Video-Swin-T* | 70.0 | 69.3 | 39.1 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EfJs5WPRKfxEvifnIO3impABNgydbiO5qqI_uCF6LYKlCQ?e=mSRLCQ) | 
| Video-Swin-T | 71.9 | 71.0 | 42.2 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/Ed3po2mJGQZHivGwMJJg8oMBumXm3Ye94oPH6wfRFK1d8A?e=NG2E9c) |
| Video-Swin-S | 72.8 | 71.5 | 42.4 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EbAiydTvu41KsMYBEFzy_d8B0Nyy1fIf2tWG7Ao-FYD0Ug?e=tmaVAu) |
| Video-Swin-B | 73.0 | 71.8 | 43.7 | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EeP1aneDDbBCo9HnMTbNjsgBpMqrgfIzJzF_jVROpZ2GWQ?e=YmkNHC) | 

\* the model is trained from scratch and set `--num_frames 6`.


### RefCOCO/+/g

We also support evaluate on RefCOCO/+/g validation set by using the pretrained weights (num_frames=1).
Specifically, we measure the P@0.5 and overall IoU (oIoU) for REC and RIS tasks, respectively.

REC (referring epression understanding):

| Backbone| RefCOCO | RefCOCO+ | RefCOCOg  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 85.0 | 79.2 | 79.0 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EepGibYBfyRGt_QedfE9SywBLF3v-bjoxo2R9E9YDqmIcw?e=7J7k1J) | 
| ResNet-101 | 85.4 | 75.8 | 79.9 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESTAK4QCkMdNkVlQz1dd7GoBo3n_i9K4_FK4YLFBAFvBrg?e=Y3PlD5) |
| Swin-T | 86.7 | 77.2 | 80.6 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESdasB6JLydDrs6mf68FrLMBuQBLBF7y_uxdveWl9oK68w?e=H5zeqk) |
| Swin-L | 89.8 | 80.0 | 83.9 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESngRLeZfV1LtrlZ7x5cVo4BR5_deWfov4Igt28LZGoDew?e=AVAsws) | 

RIS (referring image segmentation):

| Backbone| RefCOCO | RefCOCO+ | RefCOCOg  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| ResNet-50 | 71.1 | 64.1 | 64.1 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/EepGibYBfyRGt_QedfE9SywBLF3v-bjoxo2R9E9YDqmIcw?e=7J7k1J) | 
| ResNet-101 | 71.8 | 61.1 | 64.9 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESTAK4QCkMdNkVlQz1dd7GoBo3n_i9K4_FK4YLFBAFvBrg?e=Y3PlD5) |
| Swin-T | 72.9 | 62.4 | 66.1 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESdasB6JLydDrs6mf68FrLMBuQBLBF7y_uxdveWl9oK68w?e=H5zeqk) |
| Swin-L | 77.1 | 65.8 | 69.3 | [weight](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wjn922_connect_hku_hk/ESngRLeZfV1LtrlZ7x5cVo4BR5_deWfov4Igt28LZGoDew?e=AVAsws) | 


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

