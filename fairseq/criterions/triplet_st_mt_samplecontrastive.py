# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
# import numpy as np
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import \
    LabelSmoothedCrossEntropyCriterion


@register_criterion("triplet_st_mt_samplecontrastive")
class TripletSTMTSampleContrastiveCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        loss_ratio,
        contrastive_temp=0.2,
        contrastive_negs=10,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task, sentence_avg, label_smoothing,
                         ignore_prefix_size, report_accuracy)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.loss_ratio = loss_ratio
        self.contrastive_temp = contrastive_temp
        self.contrastive_negs = contrastive_negs

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0.,
                            type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means none')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--loss-ratio', default=[1, 1, 1],
                            type=float, nargs=3)
        parser.add_argument('--contrastive-temp', default=0.2, type=float)
        parser.add_argument('--contrastive-negs', default=10, type=int)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        st_net_output, audio_internal = \
            model.forward_with_internal(**sample["net_input"])
        st_loss, st_nll_loss = self.compute_loss(
            model, st_net_output, sample, reduce=reduce)

        mt_input = {
            "src_tokens": sample["src_text"],
            "src_lengths": sample["src_text_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "mask": sample["net_input"]["mask"],
        }
        mt_net_output, text_internal = model.forward_with_internal(**mt_input)
        mt_loss, mt_nll_loss = self.compute_loss(
            model, mt_net_output, sample, reduce=reduce)

        contrastive_loss = self.compute_contrastive(
            audio_internal, text_internal, reduce)

        loss = sum([self.loss_ratio[_i] * _loss for _i, _loss in
                    enumerate((st_loss, mt_loss, contrastive_loss))])
        nll_loss = sum([self.loss_ratio[_i] * _loss for _i, _loss in
                        enumerate((st_nll_loss, mt_nll_loss))])
        sample_size = (
            sample["target"].size(0) if self.sentence_avg
            else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "st_loss": st_loss.data,
            "st_nll_loss": st_nll_loss.data,
            "mt_loss": mt_loss.data,
            "mt_nll_loss": mt_nll_loss.data,
            "contrastive_loss": contrastive_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(
                model, st_net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_contrastive(self, input1, input2, reduce):
        assert input1.shape == input2.shape
        input1 = input1.transpose(0, 1)
        input2 = input2.transpose(0, 1)  # [batch, seqlen, dim]
        loss1 = self._contrastive_either_side(input1, input2, reduce)
        loss2 = self._contrastive_either_side(input2, input1, reduce)
        loss = loss1 + loss2

        return loss

    def _contrastive_either_side(self, input1, input2, reduce):
        batch_size, seqlen, dim = input1.shape
        if batch_size <= 1:
            return input1.sum() * 0
        n_negs = min(batch_size - 1, self.contrastive_negs)
        device = input1.device
        positive = input2
        negative_cands = torch.arange(batch_size-1)[None].repeat(batch_size, 1)
        offset = (negative_cands >= torch.arange(batch_size).unsqueeze(1)
                  ).long()
        negative_cands += offset
        negative_index2 = torch.randperm(batch_size - 1)[:n_negs]
        negative_index = negative_cands.index_select(
            1, negative_index2
        ).to(device)
        negative = input2.index_select(0, negative_index.view(-1)).reshape(
            batch_size, n_negs, seqlen, -1
        )
        pos_neg = torch.cat([positive.unsqueeze(1), negative], 1)
        logits = torch.cosine_similarity(
            input1.float().unsqueeze(1),
            pos_neg.float(),
            dim=-1
        ).type_as(input1)
        logits /= self.contrastive_temp
        target = torch.zeros(batch_size, seqlen, dtype=int)\
            .to(device)
        loss = F.cross_entropy(logits, target,
                               reduction='sum' if reduce else "none")
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        for name in ('loss', 'nll_loss', 'st_loss', 'st_nll_loss', 'mt_loss',
                     'mt_nll_loss', 'contrastive_loss', 'ntokens'):
            _sum = sum(log.get(name, 0) for log in logging_outputs)
            metrics.log_scalar(
                name, _sum / sample_size / math.log(2), sample_size, round=3
            )

        for name in ('', 'st_', 'mt_'):
            metrics.log_derived(
                name+"ppl",
                lambda meters: utils.get_perplexity(
                    meters[name+"nll_loss"].avg)
            )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
