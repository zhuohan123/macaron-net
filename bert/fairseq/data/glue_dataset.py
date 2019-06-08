# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad=True
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target = torch.stack([s['target'] for s in samples])
    target = target.index_select(0, sort_order)
    ntokens = sum(len(s['source']) for s in samples)

    segment = merge('segment', left_pad=left_pad)
    segment = segment.index_select(0, sort_order)

    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'segment': segment
        },
        'target': target,
        'nsentences': samples[0]['source'].size(0),
    }
    return batch


class GlueSingleDataset(FairseqDataset):
    def __init__(
        self, src, src_sizes, src_dict, labels,
        left_pad=True, max_positions=384, shuffle=True
    ):
        self.src = src
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.left_pad = left_pad
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.labels = labels

    def __getitem__(self, index):
        source = torch.cat([
            torch.LongTensor([self.src_dict.cls()]),
            data_utils.truncate_single(self.src[index], self.max_positions)
        ])
        return {
            'id': index,
            'source': source,
            'target': self.labels[index],
            'segment': torch.ones_like(source)
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), left_pad=self.left_pad
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=384):
        """Return a dummy batch with a given number of tokens."""
        src_len = utils.resolve_max_positions(
            src_len,
            max_positions,
            self.max_positions
        )
        bsz = num_tokens // src_len
        source = self.src_dict.dummy_sentence(src_len - 1)
        source = torch.cat([torch.LongTensor([self.src_dict.cls()]), source])
        return self.collater([
            {
                'id': i,
                'source': source,
                'target': torch.tensor(0),
                'segment': torch.ones_like(source)
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return min(self.src_sizes[index] + 1, self.max_positions)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self.src_sizes[index] + 1, self.max_positions)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]


class GluePairDataset(FairseqDataset):
    def __init__(
        self, src, src_sizes, src_dict, labels,
        left_pad=True, max_positions=384, shuffle=True, symmetric=False
    ):
        self.src = src
        assert len(self.src) % 2 == 0
        self.src_sizes = np.array(src_sizes)
        self.src_dict = src_dict
        self.left_pad = left_pad
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.labels = labels
        self.symmetric = symmetric

    def __getitem__(self, index):
        sent1, sent2 = self.src[index * 2], self.src[index * 2 + 1]
        if self.symmetric and np.random.rand() < 0.5:
            sent1, sent2 = sent2, sent1
        sent1, sent2 = data_utils.truncate_pair(sent1, sent2, self.max_positions)
        return {
            'id': index,
            'source': torch.cat([torch.LongTensor([self.src_dict.cls()]), sent1, sent2]),
            'target': self.labels[index],
            'segment': torch.cat([torch.LongTensor([1]), torch.ones_like(sent1), torch.zeros_like(sent2)])
        }

    def __len__(self):
        return len(self.src) // 2

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), left_pad=self.left_pad
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=384):
        """Return a dummy batch with a given number of tokens."""
        src_len = utils.resolve_max_positions(
            src_len,
            max_positions,
            self.max_positions
        )
        bsz = num_tokens // src_len
        source = self.src_dict.dummy_sentence(src_len - 1)
        source = torch.cat([torch.LongTensor([self.src_dict.cls()]), source])
        return self.collater([
            {
                'id': i,
                'source': source,
                'target': torch.tensor(0),
                'segment': torch.ones_like(source)
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return min(self.src_sizes[index * 2] + self.src_sizes[index * 2 + 1] + 1, self.max_positions)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return min(self.src_sizes[index * 2] + self.src_sizes[index * 2 + 1] + 1, self.max_positions)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        out_sizes = np.minimum(self.src_sizes[::2] + self.src_sizes[1::2] + 1, self.max_positions)
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(out_sizes, kind='mergesort')]