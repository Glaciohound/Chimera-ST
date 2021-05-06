# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging

import pickle
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)


@register_criterion("aligning_loss_bimodal")
class AligningLossBimodalCriterion(FairseqCriterion):
    def __init__(self, task, loss, fix_side=None,
                 log_keys=None):
        super().__init__(task)
        self.log_keys = [] if log_keys is None else eval(log_keys)
        self.loss = loss
        self.fix_side = fix_side
        self.fixed_model = None

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')
        parser.add_argument('--loss', type=str, default='cosine',
                            choices=['cosine', 'l2'])
        parser.add_argument('--fix-side', type=str, default=None,
                            choices=['audio', 'text'])
        # fmt: on

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'aligned_feature_fn')
        if self.fix_side is not None and self.fixed_model is None:
            self.fixed_model = pickle.loads(pickle.dumps(model)).\
                requires_grad_(False)

        net_input = sample['net_input']
        net_output = model(**net_input)
        audio_input = {
            'src_tokens': net_input['src_tokens'],
            'src_lengths': net_input['src_lengths'],
        }
        text_input = {
            'src_tokens': sample['target'],
            'src_lengths': sample['target_lengths'],
        }
        sample_size = sample['target'].numel()

        def choose_model(side):
            return self.fixed_model if side == self.fix_side else model

        audio_output = model.aligned_feature_fn(**audio_input)
        text_output = model.aligned_feature_fn(**text_input)
        if self.fix_side == 'text':
            fixed_output = self.fixed_model(**text_input)
        elif self.fix_side == 'audio':
            fixed_output = self.fixed_model(**audio_input)

        def loss_fn(x1, x2):
            if self.loss == 'cosine':
                loss = (
                    1 - F.cosine_similarity(x1, x2, -1)
                ).sum()
            elif self.loss == 'l2':
                loss = (x1 - x2).pow(2).mean(-1).sum()
            else:
                raise NotImplementedError()
            return loss

        if self.fix_side is None:
            loss = loss_fn(audio_output, text_output)
        else:
            loss = loss_fn(audio_output, fixed_output) + \
                loss_fn(text_output, audio_output)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample['id'].numel(),
            "sample_size": sample_size
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0)
                                  for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0)
                                 for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)

        # correct = sum(log.get("correct", 0) for log in logging_outputs)
        # metrics.log_scalar("_correct", correct)

        # total = sum(log.get("count", 0) for log in logging_outputs)
        # metrics.log_scalar("_total", total)

        # if total > 0:
        #     metrics.log_derived(
        #         "accuracy",
        #         lambda meters: safe_round(
        #             meters["_correct"].sum / meters["_total"].sum, 5
        #         )
        #         if meters["_total"].sum > 0
        #         else float("nan"),
        #     )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            # "correct",
            # "count",
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                val = sum(log.get(k, 0) for log in logging_outputs) / len(
                    logging_outputs
                )
                if k.startswith("loss"):
                    metrics.log_scalar(k, val / sample_size / math.log(2),
                                       sample_size)
                else:
                    metrics.log_scalar(k, val, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
