# example 
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
main_pretrain.py --with_box_refine --dataset_file all --batch_size 16 --num_frames 1 --binary  \
--resume r50_pretrained.pth  --eval
# --resume [path/to/weights.pth] --backbone [backbonetm]