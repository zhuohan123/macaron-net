# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('exp')
class ExpSchedule(FairseqLRScheduler):
    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with exp.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        self.init_lr = args.init_lr
        self.gamma = args.decay_rate_step
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--init-lr', default=0.0, type=float, metavar='LR',
                            help="initial learning rate")
        parser.add_argument('--decay-rate-step', default=1.0, type=float, metavar='GAMMA',
                            help="exponent to multiply after each step")

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        self.lr = self.init_lr * (self.gamma ** num_updates)
        self.optimizer.set_lr(self.lr)
        return self.lr
