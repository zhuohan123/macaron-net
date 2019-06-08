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

    target = merge('target', left_pad=left_pad)
    target = target.index_select(0, sort_order)
    ntokens = sum(len(s['target']) for s in samples)

    segment = merge('segment', left_pad=left_pad)
    segment = segment.index_select(0, sort_order)

    label = torch.stack([s['label'] for s in samples])
    label = label.index_select(0, sort_order)

    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'segment': segment,
            'output_mask': target.ne(pad_idx)
        },
        'target': target,
        'label': label,
        'nsentences': samples[0]['source'].size(0),
    }
    return batch


def transform(sentences, idx, dictionary, next_sent, prob, mask_prob, random_prob, max_positions, *, dummy=-1):
    # Generate NSP
    if dummy >= 0:  # generate a dummy case for testing
        dummy = (dummy - 1) // 2
        sentence = torch.cat([dictionary.dummy_sentence(dummy), dictionary.dummy_sentence(dummy)])
        segment = torch.cat([torch.ones(dummy, dtype=torch.long), torch.zeros(dummy, dtype=torch.long)])
        label = np.random.randint(0, 2)
    else:
        sent1, sent2 = data_utils.truncate_pair(sentences[idx], sentences[next_sent[idx]], max_positions)
        sentence = torch.cat([sent1, sent2])
        segment = torch.cat([torch.ones_like(sent1), torch.zeros_like(sent2)])
        label = int(next_sent[idx] == idx + 1)

    # Generate MLM
    to_mask = torch.arange(sentence.size(0))[sentence > dictionary.nspecial]
    to_mask = to_mask[torch.randperm(to_mask.size(0))[:int(round(to_mask.size(0) * prob))]]
    source = sentence.clone()
    target = torch.ones_like(sentence) * dictionary.pad()
    target[to_mask] = sentence[to_mask]
    for i in to_mask:
        r = np.random.random()
        if r < mask_prob:
            source[i] = dictionary.mask()
        elif r < mask_prob + random_prob:
            source[i] = np.random.randint(low=dictionary.nspecial, high=len(dictionary.symbols))

    # Add cls tag
    source = torch.cat([torch.LongTensor([dictionary.cls()]), source])
    target = torch.cat([torch.LongTensor([dictionary.cls()]), target])
    segment = torch.cat([torch.LongTensor([1]), segment])

    return source, target, segment, torch.tensor(label)


class BertDataset(FairseqDataset):
    def __init__(
            self, src, src_sizes, src_dict,
            left_pad=True, max_positions=384, shuffle=True,
            masked_lm_prob=0.15, mlm_mask_prob=0.8, mlm_random_prob=0.1
    ):
        self.src = src
        self.src_sizes = np.array(src_sizes)
        self.next_sent = None
        self.out_sizes = None
        self.src_dict = src_dict
        self.left_pad = left_pad
        self.max_positions = max_positions
        self.shuffle = shuffle
        self.masked_lm_prob = masked_lm_prob
        self.mlm_mask_prob = mlm_mask_prob
        self.mlm_random_prob = mlm_random_prob

    def __getitem__(self, index):
        source, target, segment, label = transform(self.src, index, self.src_dict, self.next_sent,
                                                   self.masked_lm_prob, self.mlm_mask_prob, self.mlm_random_prob,
                                                   self.max_positions)
        return {
            'id': index,
            'source': source,
            'target': target,
            'segment': segment,
            'label': label
        }

    def __len__(self):
        return len(self.src) - 1

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
        source, target, segment, label = transform(self.src, -1, self.src_dict, self.next_sent,
                                                   self.masked_lm_prob, self.mlm_mask_prob, self.mlm_random_prob,
                                                   self.max_positions, dummy=src_len)
        return self.collater([
            {
                'id': i,
                'source': source,
                'target': target,
                'segment': segment,
                'label': label
            }
            for i in range(bsz)
        ])

    def update_nsp_indices(self):
        n = len(self.src)
        perm = np.random.permutation(n - 1)
        to_roll = perm[:(n - 1) // 2]
        to_keep = perm[(n - 1) // 2:]
        self.next_sent = np.empty(n - 1, dtype=np.int64)
        self.next_sent[to_keep] = to_keep + 1
        self.next_sent[to_roll] = np.random.permutation(to_roll) + 1
        self.out_sizes = np.minimum(self.src_sizes[:-1] + self.src_sizes[self.next_sent] + 1, self.max_positions)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.out_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.out_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        self.update_nsp_indices()
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.out_sizes[indices], kind='mergesort')]
