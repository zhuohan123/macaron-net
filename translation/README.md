# Introduction

Here we present the instructions to reproduce the machine translation results from our paper. The codes are based on open-sourced [fairseq](https://github.com/pytorch/fairseq) (v0.6.0). Follow [this link](https://fairseq.readthedocs.io/) for a detailed document about the original code base and [this link](https://github.com/pytorch/fairseq/tree/v0.6.0/examples/translation) for some examples of training baseline Transformer models for machine translation with fairseq.

We also provide [pre-trained models](#pre-trained-models) for several benchmark translation datasets.

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

The scripts for training and testing Macaron Net is located at `macaron-scripts` folder. Please refer to [this page](macaron-scripts/data-preprocessing/README.md) to preprocess and get binarized data or use the data we provided in the next section. To reproduce the results by yourself:

```
# IWSLT14 De-En
## To train the model
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/train/train-iwslt14.sh
## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/test/test-iwslt14.sh checkpoint_best.pt

# WMT14 De-En base
## To train the model
$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./macaron-scripts/train/train-wmt14-base.sh
## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/test/test-wmt14-base.sh checkpoint_best.pt

# WMT14 De-En big
## To train the model
$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./macaron-scripts/train/train-wmt14-big.sh
## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/test/test-wmt14-big.sh checkpoint_best.pt
```

If you happen to have multiple nodes, you can accelerate training by using distributed scripts:
```
# WMT14 De-En big
## To train the model in a distributed manner
## Before running, updating the MASTER_HOST IP in the script and prepare a hostfile
$ mpirun --hostfile your_hostfile ./macaron-scripts/train/train-wmt14-big-distributed.sh
```

# Pre-trained Models

We provide following pre-trained models and pre-processed, binarized datasets for reproduction:

Description | Dataset | Model | Test set(s)
---|---|---|---
Macaron-net `small` | [IWSLT14 English-German](https://drive.google.com/file/d/1fBG7DmbH0luD8EKqjviG5Equgkaxv3vv/view?usp=sharing) | [download (.tbz2)](https://drive.google.com/open?id=1HnPJTxUKc6aqqXLlxOF0fourQN4K4zCo) | IWSLT14 test set (shared vocab): <br> [download (.tbz2)](https://drive.google.com/open?id=1Vza4Yh7ev1336fWpgxGalkSLhb5dHxBa)
Macaron-net `base` | [WMT14 English-German](https://drive.google.com/file/d/1iOdEGsWr5otcOOsMioVPeYEJEIyTi85J/view?usp=sharing) | [download (.tbz2)](https://drive.google.com/file/d/1EzZdueTAI-dgPGjzxXyB7fgv97JcxgE6/view?usp=sharing) | newstest2014 (shared vocab): <br> [download (.tbz2)](https://drive.google.com/file/d/1bM11V3gjKH9eWVVzrP1FEtqiGAeE8t4o/view?usp=sharing)
Macaron-net `big` | [WMT14 English-German](https://drive.google.com/file/d/1iOdEGsWr5otcOOsMioVPeYEJEIyTi85J/view?usp=sharing) | [download (.tbz2)](https://drive.google.com/file/d/1uonnRFE2ktjKTTlhgGY6sEVaKEXC9sLA/view?usp=sharing) | newstest2014 (shared vocab): <br> [download (.tbz2)](https://drive.google.com/file/d/1bM11V3gjKH9eWVVzrP1FEtqiGAeE8t4o/view?usp=sharing)

Example usage:
```
# IWSLT14 De-En
## at macaron-net/translation/, after download the tbz2 file
$ tar xf iwslt14-deen-pretrained-model.tbz2
$ CUDA_VISIBLE_DEVICES=0 ./macaron-scripts/test/test-iwslt14.sh checkpoint_pretrained.pt
...
pretrained | Generate test with beam=5: BLEU4 = 35.43, 68.6/43.2/29.3/20.3 (BP=0.973, ratio=0.973, syslen=127659, reflen=131156)
```