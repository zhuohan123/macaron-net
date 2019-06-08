import os
import sys
import copy
import itertools
import inspect

def task(name, n_sentences, task, criterion, symmetric, n_classes, data_path):
    return locals()

def params(*args):
    keys = ["seed_list", "n_epoch_list", "batch_sz_list", "lr_list", "weight_decay_list"]
    assert len(args) == len(keys)
    values = itertools.product(*args)
    return [{k: v for k, v in zip(keys, vs)} for vs in values]

cola = (
    task("cola", 8551, "glue_single", "cross_entropy_classify_binary", "", 1, "CoLA"),
    params(["100 200 300 400 500 600"], ["3 4 5"], ["16 32"], ["0.00005 0.00003"], ["0.00 0.01"])
) # 60s / epoch, 3h / search
mrpc = (
    task("mrpc", 3668, "glue_pair", "cross_entropy_classify_binary", "--symmetric", 1, "MRPC"),
    params(["100 200 300 400 500 600"], ["3 4 5"], ["16 32"], ["0.00005 0.00003"], ["0.00 0.01"])
) # 50s / epoch, 3h / search
sts = (
    task("sts", 5749, "glue_pair", "mean_squared_error", "--symmetric", 1, "STS-B"),
    params(["100 200 300 400 500 600"], ["3 4 5"], ["16 32"], ["0.00005 0.00003"], ["0.00 0.01"])
) # 50s / epoch, 4h / search
rte = (
    task("rte", 2475, "glue_pair", "cross_entropy_classify", "", 2, "RTE"),
    params(["100 200 300 400 500 600"], ["3 4 5"], ["16 32"], ["0.00005 0.00003"], ["0.00 0.01"])
) # 60s / epoch, 3h / search
mnli = (
    task("mnli", 392702, "glue_pair", "cross_entropy_classify", "", 3, "MNLI"),
    params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003"], ["0.00", "0.01"])
) # 5000s / epoch, bs 32 oom
mnlimm = (
    task("mnlimm", 392702, "glue_pair", "cross_entropy_classify", "", 3, "MNLI-mm"),
    params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003"], ["0.00", "0.01"])
) # 5000s / epoch, bs 32 oom
qnli = (
    task("qnli", 108436, "glue_pair", "cross_entropy_classify", "", 2, "QNLI-new"),
    params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003"], ["0.00", "0.01"])
) # 1600s / epoch, bs 32 oom
qqp = (
    task("qqp", 363849, "glue_pair", "cross_entropy_classify_binary", "--symmetric", 1, "QQP"),
    params(["100", "200", "300"], ["3 4 5"], ["16 24"], ["0.00005", "0.00003"], ["0.00", "0.01"])
) # 4000s / epoch, bs 32 oom
sst = (
    task("sst", 67349, "glue_single", "cross_entropy_classify", "", 2, "SST-2"),
    params(["100", "200", "300", "400", "500", "600"], ["3 4 5"], ["16 32"], ["0.00005 0.00003"], ["0.00 0.01"])
) # 400s / epoch, 18h / search

task_list = [cola, mrpc, sts, rte, mnli, mnlimm, qnli, qqp, sst]

bert_model_config = {
    "bert_model_name": "macaron_pretrained",
    "bert_model_path": "log/bert/transformer_bert_base_macaron/checkpoint_pretrained.pt",
    "bert_model_arch": "transformer_classifier_base_macaron",
}

script_dir = os.path.join("generated/", bert_model_config["bert_model_name"])

env_vars = """
PROBLEM={name}
BERT_MODEL_NAME={bert_model_name}
TASK={task}
BERT_MODEL_PATH={bert_model_path}
N_CLASSES={n_classes}
ARCH={bert_model_arch}
N_SENT={n_sentences}
CRITERION={criterion}
SYMMETRIC={symmetric}
DATA_PATH=data/glue/{data_path}
SEED_LIST="{seed_list}" 
N_EPOCH_LIST="{n_epoch_list}" 
BATCH_SZ_LIST="{batch_sz_list}" 
LR_LIST="{lr_list}" 
WEIGHT_DECAY_LIST="{weight_decay_list}" 
"""

script_template = r"""
CODE_PATH=.
cd $CODE_PATH
export PYTHONPATH=$CODE_PATH:$PYTHONPATH

for SEED in $SEED_LIST
do
    for N_EPOCH in $N_EPOCH_LIST
    do
        for BATCH_SZ in $BATCH_SZ_LIST
        do
            SENT_PER_GPU=$(( BATCH_SZ / 1 ))
            N_UPDATES=$(( ((N_SENT + BATCH_SZ - 1) / BATCH_SZ) * N_EPOCH ))
            WARMUP_UPDATES=$(( (N_UPDATES + 5) / 10 ))
            echo $SENT_PER_GPU $N_UPDATES $WARMUP_UPDATES
            for LR in $LR_LIST
            do
                for WEIGHT_DECAY in $WEIGHT_DECAY_LIST
                do

OUTPUT_PATH=log/bert_downstream/$BERT_MODEL_NAME/$PROBLEM/${N_EPOCH}-${BATCH_SZ}-${LR}-${WEIGHT_DECAY}-$SEED

mkdir -p $OUTPUT_PATH

python train.py $DATA_PATH --task $TASK --load-bert $BERT_MODEL_PATH --load-type no_out \
--arch $ARCH --n-classes $N_CLASSES \
--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 0.0 --weight-decay $WEIGHT_DECAY \
--lr $LR --lr-scheduler linear --warmup-init-lr 1e-07 --warmup-updates $WARMUP_UPDATES --min-lr 1e-09 \
--criterion $CRITERION $SYMMETRIC \
--max-sentences $SENT_PER_GPU --max-update $N_UPDATES --seed $SEED \
--save-dir $OUTPUT_PATH --no-progress-bar --log-interval 100 --no-epoch-checkpoints \
| tee -a $OUTPUT_PATH/train_log.txt

                done
            done
        done
    done
done
"""

os.makedirs(script_dir, exist_ok=True)
os.system('cp {} {}'.format(__file__, script_dir))

for task_dict, params_list in task_list:
    for i, param_dict in enumerate(params_list):
        result_dict = {}
        result_dict.update(task_dict)
        result_dict.update(bert_model_config)
        result_dict.update(param_dict)
        this_env_var = env_vars.format(**result_dict)
        script = this_env_var + script_template
        script_name = os.path.join(script_dir, ".".join([task_dict["name"], "%02d" % i, "sh"]))
        print(script_name)
        with open(script_name, "w") as f:
            f.write(script)
