# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import OrderedDict
import contextlib
import logging

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round

logger = logging.getLogger(__name__)


@register_criterion("wav2vec_bimodal")
class Wav2vecBiModalCriterion(FairseqCriterion):
    def __init__(self, task,
                 infonce=False, loss_weights=None, log_keys=None,
                 loss_weight_audio_vs_text=None,
                 l2_loss=False, mix_contrastive_method=None,
                 debug_double_audio=False, debug_audio_random_offset=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = None if loss_weights is None \
            else eval(loss_weights)
        self.log_keys = [] if log_keys is None else eval(log_keys)
        self.audio_text_loss_ratio = loss_weight_audio_vs_text
        self.l2_loss = l2_loss
        self.mix_contrastive_method = mix_contrastive_method
        self.debug_double_audio = debug_double_audio
        self.debug_audio_random_offset = debug_audio_random_offset

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--infonce', action='store_true',
                            help='uses cross entropy instead of '
                            'binary cross entropy (i.e. InfoNCE loss)')
        parser.add_argument('--loss-weights', type=str, default=None,
                            help='weights for additional loss terms')
        parser.add_argument('--log-keys', type=str, default=None,
                            help='output keys to log')
        parser.add_argument('--loss-weight-audio-vs-text',
                            type=float, nargs=4, default=[0.2, 1, 1, 1])
        parser.add_argument('--l2-loss', action='store_true', default=False)
        parser.add_argument('--debug-double-audio', action='store_true',
                            default=False)
        parser.add_argument('--debug-audio-random-offset', default=None,
                            type=int)
        parser.add_argument(
            '--mix-contrastive-method', default=None, type=str,
            choices=[
                None,
                'algebraic', 'stochastic',
                'reverse',
            ]
        )
        # fmt: on

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        w2v_model = model.encoder.wav2vec_model

        net_input = sample['net_input']
        assert 'src_tokens' in net_input
        audio_input = {
            'source': net_input['src_tokens'],
            'padding_mask': net_input.get('padding_mask', None),
            'mask': net_input['mask']
        }
        text_quasiwave, _ = \
            model.encoder._get_text_feature(
                sample['target'], sample['target_lengths']
            )
        text_input = {
            'source': text_quasiwave,
            'padding_mask': None,
            'mask': net_input['mask']
        }
        if self.debug_double_audio:
            text_input = audio_input
        audio_output, text_output, toa_output, aot_output, encout \
            = w2v_model.bimodal_forward(
                audio_input, text_input,
                mix_contrastive_method=self.mix_contrastive_method,
                debug_audio_random_offset=self.debug_audio_random_offset,
            )
        audio_encout, text_encout = encout

        with torch.no_grad() if self.l2_loss else contextlib.ExitStack():
            audio_loss, audio_sample_size, audio_logging \
                = self.process_output(
                    sample, audio_output, w2v_model, reduce, log_pred)
            loss = self.audio_text_loss_ratio[0] * audio_loss

            text_loss, text_sample_size, text_logging \
                = self.process_output(
                    sample, text_output, w2v_model, reduce, log_pred)
            loss += self.audio_text_loss_ratio[1] * text_loss

            aot_loss, aot_sample_size, aot_logging \
                = self.process_output(
                    sample, aot_output, w2v_model, reduce, log_pred)
            loss += self.audio_text_loss_ratio[2] * aot_loss

            toa_loss, toa_sample_size, toa_logging \
                = self.process_output(
                    sample, toa_output, w2v_model, reduce, log_pred)
            loss += self.audio_text_loss_ratio[3] * toa_loss

        if self.l2_loss:
            text_length = text_encout.shape[1]
            loss = (
                F.interpolate(
                    audio_encout.transpose(1, 2), size=text_length,
                    mode='linear'
                ).transpose(1, 2) -
                text_encout
            ).pow(2).sum()

        sample_size = audio_sample_size
        logging_output = OrderedDict(audio_logging)
        for _name, _loggings in zip(
            ['audio', 'text', 'aot', 'toa'],
            [audio_logging, text_logging, aot_logging, toa_logging]
        ):
            if _loggings is not None:
                for _key, _value in _loggings.items():
                    logging_output[f"{_name}_{_key}"] = _value
                if _loggings.get('count', 0) > 0:
                    logging_output[f"{_name}_accuracy"] =\
                        _loggings['correct'] / _loggings['count']

        return loss, sample_size, logging_output

    def process_output(self, sample, net_output, w2v_model,
                       reduce=True, log_pred=False):
        logits = w2v_model.get_logits(net_output).float()
        target = w2v_model.get_targets(None, net_output)

        weights = None
        if hasattr(w2v_model, "get_target_weights") and not self.infonce:
            weights = w2v_model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        if self.infonce:
            loss = F.cross_entropy(
                logits,
                target,
                reduction="sum" if reduce else "none",
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits,
                target.float(),
                weights,
                reduction="sum" if reduce else "none",
            )

        sample_size = target.numel() if self.infonce else \
            target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(w2v_model, "get_extra_losses")
            extra_losses = w2v_model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    both = max & min
                    corr = max.long().sum().item() - both.long().sum().item()
                    count = max.numel()

                logging_output["correct"] = corr
                logging_output["count"] = count
                # logging_output["accuracy"] = corr / count

        if log_pred:
            logging_output["logits"] = logits.cpu().numpy()
            logging_output["target"] = target.cpu().numpy()
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

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar("_total", total)

        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda meters: safe_round(
                    meters["_correct"].sum / meters["_total"].sum, 5
                )
                if meters["_total"].sum > 0
                else float("nan"),
            )

        builtin_keys = {
            "loss",
            "ntokens",
            "nsentences",
            "sample_size",
            "correct",
            "count",
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
