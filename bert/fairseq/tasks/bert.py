# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import numpy as np
from torch.utils.data import ConcatDataset

from fairseq import options
from fairseq.data import (
    Dictionary, BertDictionary, IndexedInMemoryDataset,
    IndexedRawTextDataset, BertDataset
)
from . import FairseqTask, register_task


@register_task('bert')
class BertTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad', default='False', type=str, metavar='BOOL',
                            help='pad the sentence on the left')
        parser.add_argument('--max-positions', default=384, type=int, metavar='N',
                            help='max number of tokens in the sequence')
        parser.add_argument('--masked-lm-prob', default=0.15, type=float, metavar='P',
                            help='masked LM probability')
        parser.add_argument('--mlm-mask-prob', default=0.8, type=float, metavar='P',
                            help='probability to mask a word in MLM')
        parser.add_argument('--mlm-random-prob', default=0.1, type=float, metavar='P',
                            help='probability to replace with a random word')

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dict = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad = options.eval_bool(args.left_pad)

        # load dictionaries
        dictionary = BertDictionary.load(os.path.join(args.data[0], 'dict.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_path):
            filename = os.path.join(data_path, split)
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path):
                return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            return None

        datasets = []

        data_paths = self.args.data

        for data_path in data_paths:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                if split_exists(split_k, data_path):
                    prefix = os.path.join(data_path, split_k)
                else:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                datasets.append(indexed_dataset(prefix, self.dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(datasets[-1])))

                if not combine:
                    break

        if len(datasets) == 1:
            dataset = datasets[0]
            sizes = dataset.sizes
        else:
            if self.args.upsample_primary > 1:
                datasets.extend([datasets[0]] * (self.args.upsample_primary - 1))
            dataset = ConcatDataset(datasets)
            sizes = np.concatenate([ds.sizes for ds in datasets])

        self.datasets[split] = BertDataset(
            dataset, sizes, self.dict, self.args.left_pad, self.args.max_positions
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_positions

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dict
