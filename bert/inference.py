#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Evaluate the perplexity of a trained language model.
"""
import os

import torch
import torch.nn.functional as F

from fairseq import options, progress_bar, tasks, utils


def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    print(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = utils.load_ensemble_for_inference(parsed_args.path.split(':'), task)

    args.__dict__.update(parsed_args.__dict__)
    print(args)

    task.args = args

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if use_cuda:
            model.cuda()
        if args.fp16:
            model.half()

    assert len(models) > 0

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens or 10000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    ).next_epoch_itr(shuffle=False)

    with progress_bar.build_progress_bar(args, itr, no_progress_bar='simple') as progress:
        with open(args.output, 'w', encoding='utf-8') as fo:
            if args.criterion == 'mean_squared_error':
                ans = torch.empty(len(task.dataset(args.gen_subset)), dtype=torch.float)
            else:
                ans = torch.empty(len(task.dataset(args.gen_subset)), dtype=torch.long)
            with torch.no_grad():
                for model in models:
                    model.eval()
                for sample in progress:
                    if use_cuda:
                        sample = utils.move_to_cuda(sample)
                    if args.criterion.endswith('binary'):
                        probs = torch.stack([F.sigmoid(model(**sample['net_input']).view(-1))
                                             for model in models], dim=0).mean(dim=0)
                        preds = torch.stack([sample['id'], (probs >= 0.5).long()], dim=1)
                    elif args.criterion == 'mean_squared_error':
                        probs = torch.stack([model(**sample['net_input']).view(-1)
                                             for model in models], dim=0).mean(dim=0)
                        preds = list(zip(*[sample['id'], probs]))
                    else:
                        probs = torch.stack([model.get_normalized_probs(model(**sample['net_input']), log_probs=False)
                                             for model in models], dim=0).mean(dim=0)
                        preds = torch.stack([sample['id'], probs.argmax(dim=-1)], dim=1)
                    for i, y in preds:
                        ans[i] = y
            for y in ans:
                fo.write(f"{y}\n")


if __name__ == '__main__':
    parser = options.get_inference_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
