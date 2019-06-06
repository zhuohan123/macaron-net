# Data Pre-processing for Neural Machine Translation

These scripts provide an example of pre-processing data for the NMT task in our paper, adapted from the [original fairseq repo](https://github.com/pytorch/fairseq/tree/v0.6.0/examples/translation).

## Preprocessing

### prepare-iwslt14.sh

Provides an example of pre-processing for IWSLT'14 German to English translation task: ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf)

Example usage for reproduction:
```bash
# Download and prepare raw data:
$ cd macaron-scripts/data-preprocessing/
$ bash prepare-iwslt14.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=macaron-scripts/data-preprocessing/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en.joined \
  --joined-dictionary
```


### prepare-wmt14en2de.sh

Provides an example of pre-processing for the WMT'14 English to German translation task. By default it will produce a dataset that was modeled after ["Attention Is All You Need" by Vaswani et al.](https://arxiv.org/abs/1706.03762) that includes news-commentary-v12 data.

Example usage for reproduction:

```bash
# Download and prepare raw data:
$ cd macaron-scripts/data-preprocessing/
$ bash prepare-wmt14en2de.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=macaron-scripts/data-preprocessing/wmt14_en_de
$ python preprocess.py --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_de_joined_dict \
  --joined-dictionary
```
