#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH
model=transformer
PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_macaron_v2
DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
OUTPUT_PATH=log/$PROBLEM/$ARCH

mkdir -p $OUTPUT_PATH

python train.py $DATA_PATH \
  --seed 1 \
  --arch $ARCH --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --dropout 0.3 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 8000 \
  --lr 0.0015 --min-lr 1e-09 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0001 \
  --max-tokens 4096 --save-dir $OUTPUT_PATH \
  --update-freq 1 --no-progress-bar --log-interval 50 \
  --ddp-backend no_c10d \
  --save-interval-updates 10000 --keep-interval-updates 20 \
| tee -a $OUTPUT_PATH/train_log.txt
