#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH

PROBLEM=iwslt14_de_en
ARCH=transformer_iwslt_de_en_macaron_v2
BEAM_SIZE=5
LPEN=1.0

DATA_PATH=data-bin/iwslt14.tokenized.de-en.joined/
MOSES_PATH=macaron-scripts/data-preprocessing/mosesdecoder
OUTPUT_PATH=log/$PROBLEM/$ARCH
TRANS_PATH=$OUTPUT_PATH/trans

CKPT=$1
CKPT_ID=$(echo $CKPT | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')
RESULT_PATH=$TRANS_PATH/$CKPT_ID/
mkdir -p $RESULT_PATH

python generate.py \
    $DATA_PATH \
    --path $OUTPUT_PATH/$CKPT \
    --batch-size 128 \
    --beam $BEAM_SIZE \
    --lenpen $LPEN \
    --remove-bpe \
    --log-format simple \
    --source-lang de \
    --target-lang en \
> $RESULT_PATH/res.txt

echo -n $CKPT_ID ""

tail -n 1 $RESULT_PATH/res.txt

