## JHMDB-Sentences

### Model Zoo

As described in the paper, we report the results using the model trained on A2D-Sentences without finetune.

| Backbone| Overall IoU | Mean IoU | mAP  | Model | 
| :----: | :----: | :----: | :----: | :----: | 
| Video-Swin-T* | 70.0 | 69.3 | 39.1 | [model](https://drive.google.com/file/d/1lK3Hulc8K6croRTrzJmz8QMhndQYDzsf/view?usp=sharing) | 
| Video-Swin-T | 71.9 | 71.0 | 42.2 | [model](https://drive.google.com/file/d/1b56oyvJZTY70ztCgDhD5vBG7fZohqzYE/view?usp=sharing) |
| Video-Swin-S | 72.8 | 71.5 | 42.4 | [model](https://drive.google.com/file/d/1edp5ZjCr4OG-8E9Co2_UgknoKMWUCB5y/view?usp=sharing) |
| Video-Swin-B | 73.0 | 71.8 | 43.7 | [model](https://drive.google.com/file/d/11UdnixY1ZQLpFqEq3_mdM4ELXGhrG7YD/view?usp=sharing) | 

\* the model is trained from scratch and set `--num_frames 6`.


### Inference & Evaluation

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file jhmdb --with_box_refine --freeze_text_encoder --batch_size 2 --resume [/path/to/model_weight] --backbone [backbone]  --eval
```

For example, evaluating the Video-Swin-Tiny model, run the following command:

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --dataset_file jhmdb --with_box_refine --freeze_text_encoder --batch_size 2 --resume a2d_video_swin_tiny.pth --backbone video_swin_t_p4w7  --eval
```
