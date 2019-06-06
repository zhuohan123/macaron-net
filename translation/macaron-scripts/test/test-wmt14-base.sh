#!/usr/bin/env bash

CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH

PROBLEM=wmt14_en_de
ARCH=transformer_wmt_en_de_macaron_v2
SRC=en
TGT=de
BEAM_SIZE=4
LPEN=0.6

DATA_PATH=data-bin/wmt14_en_de_joined_dict
TEST_DATA_PATH=$DATA_PATH/raw-test
MOSES_PATH=macaron-scripts/data-preprocessing/mosesdecoder
OUTPUT_PATH=log/$PROBLEM/$ARCH
TRANS_PATH=$OUTPUT_PATH/trans
TEST_TAG=wmt14

CKPT=$1
CKPT_ID=$(echo $CKPT | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')
RESULT_PATH=$TRANS_PATH/$CKPT_ID/
mkdir -p $RESULT_PATH

python interactive.py \
    $DATA_PATH \
    --path $OUTPUT_PATH/$CKPT \
    --batch-size 128 \
    --beam $BEAM_SIZE \
    --lenpen $LPEN \
    --remove-bpe \
    --log-format simple \
    --buffer-size 12800 \
    --source-lang en \
    --target-lang de \
< $TEST_DATA_PATH/en-de.en.bpe \
> $RESULT_PATH/res.txt

cat $RESULT_PATH/res.txt | awk -F '\t' '/^H\t/ {print $3}' > $RESULT_PATH/hyp.txt
cat $RESULT_PATH/hyp.txt | perl $MOSES_PATH/scripts/tokenizer/detokenizer.perl -q -threads 8 -a -l $TGT > $RESULT_PATH/hyp.detok.txt
cat $RESULT_PATH/hyp.detok.txt | perl $MOSES_PATH/scripts/tokenizer/tokenizer.perl -l $TGT > $RESULT_PATH/hyp.tok.txt
cat $RESULT_PATH/hyp.tok.txt | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $RESULT_PATH/hyp.tok.atat.txt
perl $MOSES_PATH/scripts/generic/multi-bleu.perl $TEST_DATA_PATH/en-de.de.tok.atat < $RESULT_PATH/hyp.tok.atat.txt > $RESULT_PATH/bleu.txt
# cat $RESULT_PATH/hyp.detok.txt | sacrebleu -t $TEST_TAG -l $SRC-$TGT --width 2 > $RESULT_PATH/bleu.txt
echo -n $CKPT_ID ""
cat $RESULT_PATH/bleu.txt

