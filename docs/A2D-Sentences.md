## A2D-Sentences

### Model Zoo

The pretrained models are the same as those provided for Ref-Youtube-VOS.

| Backbone| Overall IoU | Mean IoU | mAP  | Pretrain | Model |
| :----: | :----: | :----: | :----: | :----: | :----: |
| Video-Swin-T* | 72.3 | 64.1 | 48.6 | - | [model](https://drive.google.com/file/d/1z-HO71IcFOZ9A6KD71wAXkbiQgKDpSp7/view?usp=sharing) \| [log](https://drive.google.com/file/d/1JhsXgcWOYv97u6tpAUnBi9-D3mxcHXzO/view?usp=sharing) |
| Video-Swin-T | 77.6 | 69.6 | 52.8 | [weight](https://drive.google.com/file/d/1g9Dm1vLdwpwSKVtIZzWKPUk2-zK3IbQa/view?usp=sharing) | [model](https://drive.google.com/file/d/1z-HO71IcFOZ9A6KD71wAXkbiQgKDpSp7/view?usp=sharing) \| [log](https://drive.google.com/file/d/1xjevouL3a1gHZN5KHtA07Cpa07R4T1Qi/view?usp=sharing) |
| Video-Swin-S | 77.7 | 69.8 | 53.9 | [weight](https://drive.google.com/file/d/1GrhFhsUidsVs7-dhY8NkVgWfBZdeit9C/view?usp=sharing) | [model](https://drive.google.com/file/d/1ng2FAX9J4FyQ7Bq1eeQC9Vvv1W8JZmek/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Uu72THexbtEje4aKXR7Q2Yd4zyPmQsi3/view?usp=sharing) |
| Video-Swin-B | 78.6 | 70.3 | 55.0 | [weight](https://drive.google.com/file/d/1MJ1362zjqu-uZdXsSQH6pI1QOFqwv5lY/view?usp=sharing) | [model](https://drive.google.com/file/d/1WlNjKS_Li-1KoUzuPM4MRM4b-oK2Ka7c/view?usp=sharing) \| [log](https://drive.google.com/file/d/1tH-f9_U0gY-iNfXm6GRyttJp3uvm5NQw/view?usp=sharing) |

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



