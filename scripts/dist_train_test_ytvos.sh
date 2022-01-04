#!/usr/bin/env bash
set -x

GPUS=${GPUS:-8}
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=$1
PRETRAINED_WEIGHTS=$2
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main.py --with_box_refine --binary --freeze_text_encoder \
--epochs 6 --lr_drop 3 5 \
--output_dir=${OUTPUT_DIR} --pretrained_weights=${PRETRAINED_WEIGHTS} ${PY_ARGS}

# inference
CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"


# ./scripts/hdfs_dist_load_pretrain_train_test_ytvos.sh pretrain_work_dirs/r50_binary_freezetext \
# pretrain_dirs/r50_deform_roberta_pretrain/checkpoint.pth \
# --binary --freeze_text_encoder --visualize \
# --epochs 6 --lr_drop 3 5 
# [backbone]
# or 
# ./scripts/hdfs_dist_load_pretrain_train_test_ytvos.sh pretrain_work_dirs/r50_binary_freezetext \
# hdfs://haruna/home/byte_arnold_hl_vc/user/wujiannan/rvosnet/pretrain_dirs/r50_deform_roberta_refcoco_epoch12/checkpoint.pth \
# --binary --freeze_text_encoder --visualize \
# --epochs 6 --lr_drop 3 5 
# [backbone]
