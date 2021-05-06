# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from copy import deepcopy
import numpy as np

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import \
    LabelSmoothedCrossEntropyCriterion


@register_criterion("triplet_st_mt_contrastive")
class TripletSTMTContrastiveCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        loss_ratio,
        contrastive_temp=0.1,
        ignore_prefix_size=0,
        report_accuracy=False,
        contrastive_increase_until=None,
        kd_ratio=None,
    ):
        super().__init__(task, sentence_avg, label_smoothing,
                         ignore_prefix_size, report_accuracy)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.loss_ratio = loss_ratio
        self.contrastive_temp = contrastive_temp
        self.contrastive_increase_until = contrastive_increase_until
        self.kd_ratio = kd_ratio

    @staticmethod
    def get_num_updates():
        return metrics.get_smoothed_values("train").get("num_updates", 0)

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
        parser.add_argument('--contrastive-temp', default=0.1,
                            type=float)
        parser.add_argument('--contrastive-increase-until', type=int,
                            default=None)
        parser.add_argument('--kd-ratio', default=[None, None],
                            type=float, nargs=2)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert not (self.kd_ratio != [None, None] and
                    self.loss_ratio != [1, 0, 0])
        st_net_output, audio_internal = \
            model.forward_with_internal(**sample["net_input"])
        if self.kd_ratio == [None, None]:
            st_loss, st_nll_loss = self.compute_loss(
                model, st_net_output, sample, reduce=reduce)
        else:
            st_loss, st_nll_loss = self.compute_loss(
                model, st_net_output, sample, reduce=False)
            batch_size = sample['nsentences']
            is_teacher = np.mod(sample['id'].cpu().data.numpy(), 2)
            kd_loss_ratio = is_teacher * self.kd_ratio[1] + \
                (1-is_teacher) * self.kd_ratio[0]
            kd_loss_ratio = \
                torch.tensor(kd_loss_ratio).to(st_loss).unsqueeze(-1)
            st_loss = st_loss.view(batch_size, -1) * kd_loss_ratio
            st_nll_loss = st_nll_loss.view(batch_size, -1) * kd_loss_ratio
            if reduce:
                st_loss = st_loss.sum()
                st_nll_loss = st_nll_loss.sum()

        if self.loss_ratio[1] != 0:
            mt_input = {
                "src_tokens": sample["src_text"],
                "src_lengths": sample["src_text_lengths"],
                "prev_output_tokens":
                sample["net_input"]["prev_output_tokens"],
                "mask": sample["net_input"]["mask"],
            }
            mt_net_output, text_internal = \
                model.forward_with_internal(**mt_input)
            mt_loss, mt_nll_loss = self.compute_loss(
                model, mt_net_output, sample, reduce=reduce)
        else:
            mt_loss, mt_nll_loss = 0, 0

        if self.loss_ratio[2] != 0:
            contrastive_loss = self.compute_contrastive(
                audio_internal, text_internal, reduce)
        else:
            contrastive_loss = 0

        loss_ratio = deepcopy(self.loss_ratio)
        if self.contrastive_increase_until is not None:
            num_updates = self.get_num_updates()
            num_updates = num_updates or 0
            loss_ratio[2] *= min(
                1, num_updates / self.contrastive_increase_until)
        loss = sum([loss_ratio[_i] * _loss for _i, _loss in
                    enumerate((st_loss, mt_loss, contrastive_loss))])
        nll_loss = sum([loss_ratio[_i] * _loss for _i, _loss in
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
            "mt_loss": mt_loss.data if self.loss_ratio[1] != 0 else 0,
            "mt_nll_loss": mt_nll_loss.data if self.loss_ratio[1] != 0 else 0,
            "contrastive_loss": contrastive_loss.data
            if self.loss_ratio[2] != 0 else 0,
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
        batch_size, seqlen, _ = input1.shape
        logits = torch.cosine_similarity(
            input1.float().unsqueeze(2),
            input2.float().unsqueeze(1),
            dim=-1
        ).type_as(input1)
        logits /= self.contrastive_temp
        target = torch.arange(seqlen)[None].repeat(batch_size, 1)\
            .to(logits.device)
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
