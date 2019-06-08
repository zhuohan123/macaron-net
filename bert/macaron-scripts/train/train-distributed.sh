#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH
model=transformer
PROBLEM=bert
ARCH=transformer_bert_base_macaron
# Because of copyright, we cannot provide our binarized data.
# Please process your own training data.
DATA_PATH=data-bin/bert_corpus/
OUTPUT_PATH=log/$PROBLEM/ARCH

mkdir -p $OUTPUT_PATH

# Example usage with 8 * 4 = 32 P40 GPUs. Change the --max-tokens and --update-freq to match your hardware settings.

MASTER_HOST="0.0.0.0" # Replace it with your master's IP

python distributed_train.py $DATA_PATH \
  --distributed-init-method tcp://$MASTER_HOST:23456 \
  --distributed-world-size $OMPI_COMM_WORLD_SIZE \
  --distributed-rank $OMPI_COMM_WORLD_RANK \
  --device-id $OMPI_COMM_WORLD_LOCAL_RANK \
  --task bert --seed 1 \
  --arch $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay 0.01 \
  --lr 0.0003 --lr-scheduler linear --warmup-updates 1 --min-lr 1e-09 \
  --criterion cross_entropy_bert \
  --max-tokens 4000 \
   --update-freq 1 --max-update 800000 --seed 3 \
  --ddp-backend no_c10d \
  --save-dir $OUTPUT_PATH --no-progress-bar --log-interval 50 --save-interval-updates 10000 --keep-interval-updates 20 \
| tee -a $OUTPUT_PATH/train_log.txt