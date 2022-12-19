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
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this


# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main_pretrain.py --with_box_refine --dataset_file all  --binary \  # do not freeze text encoder
--batch_size 2 --num_frames 1 \
--epochs 12 --lr_drop 8 10 \
--output_dir=${OUTPUT_DIR} ${PY_ARGS}
# --backbone [backbone]

echo "Working path is: ${OUTPUT_DIR}"

