#!/usr/bin/env bash

BERT_DIR=log/bert/transformer_bert_base_macaron
CKPT=$1
CKPT_ID=$(echo $CKPT | sed 's/checkpoint//g' | sed 's/\.pt//g' | sed 's/^_//g')
BERT_PATH=${BERT_DIR}/$CKPT
DATA_PATH=glue

function run_exp {
    TASK_NAME=$1
    TASK_TYPE=$2
    SYMMETRIC_FLAG=$3
    TASK_CRITERION=$4
    N_CLASSES=$5
    N_SENT=$6
    WEIGHT_DECAY=$7
    N_EPOCH=$8
    BATCH_SZ=$9
    LR=${10}
    SEED=${11}

    # Runs on 1 GPU
    SENT_PER_GPU=$(( BATCH_SZ / 1 ))
    N_UPDATES=$(( ((N_SENT + BATCH_SZ - 1) / BATCH_SZ) * N_EPOCH ))
    WARMUP_UPDATES=$(( (N_UPDATES + 5) / 10 ))

    mkdir -p ${BERT_DIR}/${CKPT_ID}/${TASK_NAME}
    python train.py data-bin/${DATA_PATH}/${TASK_NAME} --task ${TASK_TYPE} ${SYMMETRIC_FLAG} \
    --arch transformer_classifier_base_macaron --n-classes ${N_CLASSES} --load-bert ${BERT_PATH} \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay ${WEIGHT_DECAY} \
    --lr ${LR} --lr-scheduler linear --warmup-init-lr 1e-07 --warmup-updates ${WARMUP_UPDATES} --min-lr 1e-09 \
    --criterion ${TASK_CRITERION} \
    --max-sentences ${SENT_PER_GPU} --max-update ${N_UPDATES} --seed ${SEED} \
    --save-dir ${BERT_DIR}/${CKPT_ID}/${TASK_NAME} --no-progress-bar --no-epoch-checkpoints

    python inference.py data-bin/${DATA_PATH}/${TASK_NAME} --gen-subset test --task ${TASK_TYPE} \
    --path ${BERT_DIR}/${CKPT_ID}/${TASK_NAME}/checkpoint_last.pt --output ${BERT_DIR}/${CKPT_ID}/prediction_${TASK_NAME}.txt
}

echo 'To reproduce our result, please run in 1 GPU'

run_exp 'CoLA' 'glue_single' '' 'cross_entropy_classify_binary' 1 8551 0.00 5 32 0.00003 400

run_exp 'MRPC' 'glue_pair' '--symmetric' 'cross_entropy_classify_binary' 1 3668 0.00 4 16 0.00005 500

run_exp 'STS-B' 'glue_pair' '--symmetric' 'mean_squared_error' 1 5749 0.00 5 16 0.00005 500

run_exp 'RTE' 'glue_pair' '' 'cross_entropy_classify' 2 2475 0.00 4 16 0.00005 200

run_exp 'SST-2' 'glue_single' '' 'cross_entropy_classify' 2 67349 0.00 3 24 0.00005 200

run_exp 'MNLI' 'glue_pair' '' 'cross_entropy_classify' 3 392702 0.00 3 24 0.00005 300

run_exp 'MNLI-mm' 'glue_pair' '' 'cross_entropy_classify' 3 392702 0.00 3 16 0.00005 300

run_exp 'QQP' 'glue_pair' '--symmetric' 'cross_entropy_classify_binary' 1 363849 0.00 5 16 0.00005 200

run_exp 'QNLI' 'glue_pair' '' 'cross_entropy_classify' 2 108436 0.01 4 16 0.00003 400

python inference.py data-bin/${DATA_PATH}/diagnostic --gen-subset test --task glue_pair \
--path ${BERT_DIR}/${CKPT_ID}/MNLI/checkpoint_last.pt --output ${BERT_DIR}/${CKPT_ID}/prediction_diagnostic.txt

mkdir -p predictions
python examples/glue/process_predictions.py predictions --output predictions
zip predictions.zip predictions/*.tsv