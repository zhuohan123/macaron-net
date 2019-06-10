# macaron-net

This repo contains the **codes** and **pretrained models** for [our paper](https://arxiv.org/pdf/1906.02762):

> Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View  
> Yiping Lu*, Zhuohan Li*, Di He, Zhiqing Sun, Bin Dong, Tao Qin, Liwei Wang, Tie-Yan Liu

The two sub-directories includes reproducible codes, pre-trained models and instructions for the machine translation and unsupervised pretraining ([BERT](https://github.com/google-research/bert)) tasks. Please find the READMEs in the sub-directories for the detailed instructions for reproduction.

Both implementations are based on open-sourced [fairseq](https://github.com/pytorch/fairseq) (v0.6.0). The codes for unsupervised pretraining tasks are based on [StackingBERT](https://github.com/gonglinyuan/StackingBERT). Note that currently the codes in `bert` subdirectories cannot be used to train translation models. We are working on merging two code bases and planning to release the unified version in the near future.

## Citation
~~~
@article{lu2019understanding,
  title={Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View},
  author={Lu, Yiping and Li, Zhuohan and He, Di and Sun, Zhiqing and Dong, Bin and Qin, Tao and Wang, Liwei and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:1906.02762},
  year={2019}
}
~~~
