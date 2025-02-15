## Ref-DAVIS17

### Model Zoo

As described in the paper, we report the results using the model trained on Ref-Youtube-VOS without finetune.

| Backbone| J&F | J | F | Model | 
| :----: | :----: | :----: | :----: | :----: |
| ResNet-50 | 58.5 | 55.8 | 61.3 | [model](https://drive.google.com/file/d/1TBykzTRiOEbpM2YEYnDICyqjwkvlD2cR/view?usp=sharing) |
| Swin-L | 60.5 | 57.6 | 63.4 | [model](https://drive.google.com/file/d/103N96BbQnT9P57aGx7TCsZVyGr3ejFdl/view?usp=sharing) |
| Video-Swin-B | 61.1 | 58.1 | 64.1 | [model](https://drive.google.com/file/d/1_dS8hyZJFdijvsDVmoVcHjWhRE6sAXvS/view?usp=sharing) |


### Inference & Evaluation

```
./scripts/dist_test_davis.sh [/path/to/output_dir] [/path/to/model_weight] --backbone [backbone]
```

For example, evaluating the Swin-Large model, run the following command:

```
./scripts/dist_test_davis.sh davis_dirs/swin_large ytvos_swin_large.pth --backbone swin_l_p4w7
```
