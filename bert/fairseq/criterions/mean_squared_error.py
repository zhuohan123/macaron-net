# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('mean_squared_error')
class MeanSquaredErrorCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input']).view(-1)  # B
        target = model.get_targets(sample, net_output).float()  # B
        loss = F.mse_loss(net_output, target, reduction='sum' if reduce else 'none')
        sample_size = target.size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'x': net_output.detach().cpu().numpy(),
            'y': target.detach().cpu().numpy()
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        # x = np.concatenate([log.get('x', np.array([])) for log in logging_outputs])
        # y = np.concatenate([log.get('y', np.array([])) for log in logging_outputs])
        # pearson = stats.pearsonr(x, y)[0]
        # spearman = stats.spearmanr(x, y)[0]
        agg_output = {
            'loss': loss_sum / sample_size,
            # 'acc': 0.5 * (pearson + spearman),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'x': np.concatenate([log.get('x', np.array([])) for log in logging_outputs]),
            'y': np.concatenate([log.get('y', np.array([])) for log in logging_outputs])
            # 'pearson': pearson,
            # 'spearman': spearman
        }
        return agg_output
