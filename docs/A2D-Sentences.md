## A2D-Sentences

### Model Zoo

The pretrained models are the same as those provided for Ref-Youtube-VOS.

| Backbone| Overall IoU | Mean IoU | mAP  | Pretrain | Model |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Video-Swin-T* | 72.3 | 64.1 | 48.6 | - | [model](https://drive.google.com/file/d/1lK3Hulc8K6croRTrzJmz8QMhndQYDzsf/view?usp=sharing) \| [log](https://drive.google.com/file/d/1_uzt6iI6TfM3cQbIblOI5_siFDxCa4mj/view?usp=sharing) |
| Video-Swin-T | 77.6 | 69.6 | 52.8 | [weight](https://drive.google.com/file/d/1qaEuOocLOZoj89unhZP3oN708ofmqopS/view?usp=sharing) | [model](https://drive.google.com/file/d/1b56oyvJZTY70ztCgDhD5vBG7fZohqzYE/view?usp=sharing) \| [log](https://drive.google.com/file/d/1n_Swnok9I2uaZtCHpBzAC53J3zILh0Rk/view?usp=sharing) |
| Video-Swin-S | 77.7 | 69.8 | 53.9 | [weight](https://drive.google.com/file/d/1SHzlzxE-KQeoGoYucdnPZtw_uz2ymgxn/view?usp=sharing) | [model](https://drive.google.com/file/d/1edp5ZjCr4OG-8E9Co2_UgknoKMWUCB5y/view?usp=sharing) \| [log](https://drive.google.com/file/d/1sOXhiCILSJTAiPYixyEQZLInzLfvFHXb/view?usp=sharing) |
| Video-Swin-B | 78.6 | 70.3 | 55.0 | [weight](https://drive.google.com/file/d/1gbyknvuOiKpxK1kljxFnt1k5Jxp3R1jv/view?usp=sharing) | [model](https://drive.google.com/file/d/11UdnixY1ZQLpFqEq3_mdM4ELXGhrG7YD/view?usp=sharing) \| [log](https://drive.google.com/file/d/1UcSGzKKDHoClHv87tg4sQi4sbH4Vr01j/view?usp=sharing) |

\* the model is trained from scratch and set `--num_frames 6`.


### Inference & Evaluation

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 2 --resume [/path/to/model_weight] --backbone [backbone]  --eval
```

For example, evaluating the Video-Swin-Tiny model, run the following command:

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 2 --resume a2d_video_swin_tiny.pth --backbone video_swin_t_p4w7  --eval
```

### Training

- Finetune

```
./scripts/dist_train_a2d.sh [/path/to/output_dir] [/path/to/pretrained_weight] --backbone [backbone]
```

For example, training the Video-Swin-Tiny model, run the following command:
```
./scripts/dist_train_a2d.sh a2d_dirs/video_swin_tiny pretrained_weights/video_swin_tiny_pretrained.pth --backbone video_swin_t_p4w7
```

- Train from scratch

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --epochs 12 --lr_drop 8 10 --dropout 0 --weight_decay 1e-4 --output_dir=[/path/to/output_dir] --backbone [backbone] --backbone_pretrained [/path/to/pretrained backbone weight] [other args]
```

For example, training the Video-Swin-Tiny model from scratch and set window size as 6, run the following command:

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --epochs 12 --lr_drop 8 10 --dropout 0 --weight_decay 1e-4 --output_dir a2d_dirs/video_swin_tiny_scratch_frame6 --backbone video_swin_t_p4w7 --bacbkone_pretrained video_swin_pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth --num_frames 6
```



