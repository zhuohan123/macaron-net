# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn.functional as F

from fairseq import utils
from . import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_classify_binary')
class CrossEntropyClassifyBinaryCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input']).view(-1)  # B
        target = model.get_targets(sample, net_output)  # B
        loss = F.binary_cross_entropy_with_logits(net_output, target.float(), reduction='sum' if reduce else 'none')
        tp = ((net_output >= 0) & (target == 1)).long().sum()
        fp = ((net_output >= 0) & (target == 0)).long().sum()
        fn = ((net_output < 0) & (target == 1)).long().sum()
        tn = ((net_output < 0) & (target == 0)).long().sum()
        assert (tp + fp + tn + fn) == target.size(0), 'invalid size'
        sample_size = target.size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'tp': utils.item(tp.data) if reduce else tp.data,
            'fp': utils.item(fp.data) if reduce else fp.data,
            'fn': utils.item(fn.data) if reduce else fn.data,
            'tn': utils.item(tn.data) if reduce else tn.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        tp_sum = sum(log.get('tp', 0) for log in logging_outputs)
        fp_sum = sum(log.get('fp', 0) for log in logging_outputs)
        fn_sum = sum(log.get('fn', 0) for log in logging_outputs)
        tn_sum = sum(log.get('tn', 0) for log in logging_outputs)
        assert tp_sum + fp_sum + fn_sum + tn_sum == sample_size, 'invalid size when aggregating'
        acc = (tp_sum + tn_sum) / sample_size
        # tmp = 2 * tp_sum + fp_sum + fn_sum
        # f1 = (2 * tp_sum) / tmp if tmp else 0
        # tmp = (tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum)
        # mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / (tmp ** 0.5) if tmp else 0
        agg_output = {
            'loss': loss_sum / sample_size,
            'acc': acc,
            'tp': tp_sum,
            'fp': fp_sum,
            'fn': fn_sum,
            'tn': tn_sum,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
