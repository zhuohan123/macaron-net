# Introduction

Here we present the instructions to reproduce the unsupervised pretraining results from our paper. The codes are based on open-sourced [fairseq](https://github.com/pytorch/fairseq) (v0.6.0) and the [StackingBERT](https://github.com/gonglinyuan/StackingBERT) repo. 

For any issues related to data pre-processing and BERT baselines, please take a look at [StackingBERT](https://github.com/gonglinyuan/StackingBERT) repo.

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

The fairseq library we use requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```

# Reproduction

The scripts for training and testing Macaron Net is located at `macaron-scripts` folder.

Because of copyright issue, we cannot provide the complete training data here. Please refer to [this repo](https://github.com/soskek/bookcorpus) to crawl the BookCorpus part of data. We also provide a pretrained model for reproduction of GLUE results.

To reproduce the results by yourself:

```
# Clone some tools
$ cd macaron-scripts/common/
$ ./clone-repos.sh

# Get binarized training data
## Please clone your own BookSorpus data before using the following script.
$ cd macaron-scripts/bert/
$ ./process_bert.sh

# Train the model
$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./macaron-scripts/train/train.sh

# Get binarized GLUE data
$ cd macaron-scripts/glue/
$ ./process_glue.sh data-bin/bert_corpus/bpe-code data-bin/bert_corpus/dict.txt

# Train & inference on GLUE
## Please refer to macaron-scripts/test/generate_test_scripts.py to do a hyperparameter search. 
## The following script uses our searched results.
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/test/test-our-best-setting.sh checkpoint_last.pt
```

If you happen to have multiple nodes, you can accelerate training by using distributed scripts:
```
$ mpirun --hostfile your_hostfile ./macaron-scripts/train/train-distributed.sh
```

# Pre-trained Models

We provide following pre-trained models and pre-processed binarized GLUE datasets for reproduction:

Description | Training Dataset | Model | Test Dataset
---|---|---|---
Macaron-net `base` | Wikipedia + BookCorpus | [download (.tbz2)](https://drive.google.com/file/d/16zBuwzlkL2tTjzJvHX7dUV66fqtoyakc/view?usp=sharing) | GLUE: <br> [download (.tbz2)](https://drive.google.com/file/d/1m_ytWY_zv62Gorgu77Uu6R0lOfbxz-7a/view?usp=sharing)

Note that our binarized GLUE dataset might eventually be outdated. Please refer to [GLUE website](https://gluebenchmark.com/tasks) for latest data.

Example usage:
```
# Test on GLUE
## at macaron-net/bert/, after download the tbz2 file
$ tar xf unsupervised-bert-base-pretrained-model.tbz2
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/test/test-our-best-setting.sh checkpoint_pretrained.pt
...
```