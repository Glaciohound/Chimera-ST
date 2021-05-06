# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
# import math
# from typing import List, Tuple

import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F
# from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import (
    # BaseFairseqModel,
    register_model, register_model_architecture
)
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model, TransformerEncoder
from fairseq.modules import (
    # Fp32GroupNorm,
    # Fp32LayerNorm,
    GradMultiply,
    # GumbelVectorQuantizer,
    # LayerNorm,
    # MultiheadAttention,
    # SamePad,
    # TransposeLast,
)
# from fairseq.modules.transformer_sentence_encoder import init_bert_params
# from fairseq.utils import buffered_arange

logger = logging.getLogger(__name__)


@register_model("wav2vec2_quasiwave")
class Wav2Vec2QuasiwaveModel(Wav2Vec2Model):
    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        return cls(args)

    def __init__(self, args):
        super().__init__(args)
        self.encoder = TransformerQuasiwaveEncoder(args)

    def bimodal_forward(
        self,
        input_dict1, input_dict2,
        features_only=False,
        mix_contrastive_method=None,
        debug_audio_random_offset=None,
    ):
        ''' both representations will be cast to length of input_dict1 '''
        features_keys = [
            'x', 'y', 'padding_mask',
            'features_pen', 'unmasked_features', 'mask_indices',
            'num_vars', 'code_ppl', 'prob_ppl', 'curr_temp'
        ]
        features1 = dict(zip(
            features_keys,
            self._extract_initial_features(**input_dict1)
        ))
        features2 = dict(zip(
            features_keys,
            self._extract_initial_features(
                **input_dict2,
                cast_feature_to_length=features1['x'].shape[1],
                mask_indices=features1['mask_indices'])
        ))
        assert (features1['mask_indices'] == features2['mask_indices']).all()
        x1 = features1['x']
        x2 = features2['x']

        if features_only:
            return {
                1: {
                    'x': features1['x'],
                    'padding_mask': features1['padding_mask']
                },
                2: {
                    'x': features2['x'],
                    'padding_mask': features2['padding_mask']
                },
            }

        # get contrastive targets
        if debug_audio_random_offset is None or debug_audio_random_offset == 0:
            x1 = x1[features1['mask_indices']].view(
                x1.size(0), -1, x1.size(-1))
        else:
            _mask_row, _mask_column = torch.where(features1['mask_indices'])
            _mask_column = (
                _mask_column + torch.randint(
                    -debug_audio_random_offset, debug_audio_random_offset,
                    size=_mask_column.shape
                ).to(_mask_column)
            ).clip(0, x1.shape[1]-1)
            x1 = x1[_mask_row, _mask_column].view(x1.size(0), -1, x1.size(-1))
        x1 = self.final_proj(x1)
        x2 = x2[features2['mask_indices']].view(x2.size(0), -1, x2.size(-1))
        x2 = self.final_proj(x2)
        contrastive_negs_keys = [
            'y', 'negs', 'num_vars', 'code_ppl', 'prob_ppl', 'curr_temp'
        ]
        contrastive_negs1 = dict(zip(
            contrastive_negs_keys,
            self._get_contrastive_negs(
                features1['y'], features1['unmasked_features'],
                features1['num_vars'], features1['code_ppl'],
                features1['prob_ppl'], features1['curr_temp'],
            )
        ))
        contrastive_negs2 = dict(zip(
            contrastive_negs_keys,
            self._get_contrastive_negs(
                features2['y'], features2['unmasked_features'],
                features2['num_vars'], features2['code_ppl'],
                features2['prob_ppl'], features2['curr_temp'],
            )
        ))

        # mutual contrastive prediction
        # length1 = x1.shape[1]
        # length2 = x2.shape[1]
        preds_1 = self.compute_preds(
            x1, contrastive_negs1['y'], contrastive_negs1['negs']
        )
        preds_2 = self.compute_preds(
            x2, contrastive_negs2['y'], contrastive_negs2['negs']
        )
        if mix_contrastive_method is None:
            preds_2on1 = self.compute_preds(
                x1,
                contrastive_negs2['y'], contrastive_negs2['negs']
            )
            preds_1on2 = self.compute_preds(
                x2,
                contrastive_negs1['y'], contrastive_negs1['negs']
            )
        elif mix_contrastive_method == 'reverse':
            preds_2on1 = self.compute_preds(
                x1,
                contrastive_negs2['y'], contrastive_negs1['negs']
            )
            preds_1on2 = self.compute_preds(
                x2,
                contrastive_negs1['y'], contrastive_negs2['negs']
            )
        elif mix_contrastive_method == 'algebraic':
            preds_2on1 = self.compute_preds(
                x1,
                contrastive_negs2['y'],
                (contrastive_negs1['negs'] + contrastive_negs2['negs']) / 2
            )
            preds_1on2 = self.compute_preds(
                x2,
                contrastive_negs1['y'],
                (contrastive_negs1['negs'] + contrastive_negs2['negs']) / 2
            )
        elif mix_contrastive_method == 'stochastic':
            preds_2on1 = self.compute_preds(
                x1,
                contrastive_negs2['y'],
                torch.cat(
                    (contrastive_negs1['negs'], contrastive_negs2['negs']), 0
                )[torch.randperm(2 * self.n_negatives)[:self.n_negatives]]
            )
            preds_1on2 = self.compute_preds(
                x2,
                contrastive_negs1['y'],
                torch.cat(
                    (contrastive_negs1['negs'], contrastive_negs2['negs']), 0
                )[torch.randperm(2 * self.n_negatives)[:self.n_negatives]]
            )
        else:
            raise NotImplementedError()

        def _gather_results(_x, _features, _contrastive_negs):
            _result = {
                'x': _x,
                'padding_mask': _features['padding_mask'],
                'features_pen': _features['features_pen'],
            }
            if _contrastive_negs['prob_ppl'] is not None:
                _result["prob_perplexity"] = _contrastive_negs['prob_ppl']
                _result["code_perplexity"] = _contrastive_negs['code_ppl']
                _result["num_vars"] = _contrastive_negs['num_vars']
                _result["temp"] = _contrastive_negs['curr_temp']
            return _result

        result = (
            _gather_results(preds_1, features1, contrastive_negs1),
            _gather_results(preds_2, features2, contrastive_negs2),
            _gather_results(preds_2on1, features2, contrastive_negs2),
            _gather_results(preds_1on2, features1, contrastive_negs1),
            (x1, x2)
        )

        return result

    def forward_clone(
            self, source, padding_mask=None, mask=None, features_only=False):
        """ just a dumplicate of original forward() function """

        # get raw and contextualized features
        (
            x, y, padding_mask,
            features_pen, unmasked_features, mask_indices,
            num_vars, code_ppl, prob_ppl, curr_temp
        ) = self._extract_initial_features(source, padding_mask, mask)
        # x : contextualized features
        # y : raw features, later used as contrastive targets, probably masked
        # unmasked_features: raw features, similar to `y`

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        # get contrastive targets based on raw masked/unmasked features
        x = x[mask_indices].view(x.size(0), -1, x.size(-1))
        x = self.final_proj(x)
        (
            y, negs, num_vars, code_ppl, prob_ppl, curr_temp
        ) = self._get_contrastive_negs(
            y, unmasked_features, num_vars, code_ppl, prob_ppl, curr_temp
        )

        # contrastive prediction
        x = self.compute_preds(x, y, negs)

        result = {"x": x,
                  "padding_mask": padding_mask,
                  "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def _extract_initial_features(self, source, padding_mask=None, mask=None,
                                  cast_feature_to_length=None,
                                  mask_indices=None):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if cast_feature_to_length is not None:
            features = F.interpolate(
                features.transpose(1, 2), size=cast_feature_to_length,
                mode='linear'
            ).transpose(1, 2)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(
                padding_mask.size(0), features.size(1), -1
            )
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask_quasiwave(
                features, padding_mask, mask_indices)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x, padding_mask = self.encoder(x, padding_mask=padding_mask)
        return (
            x, y, padding_mask,
            features_pen, unmasked_features, mask_indices,
            num_vars, code_ppl, prob_ppl, curr_temp
        )

    def _get_contrastive_negs(
        self, y, unmasked_features,
        num_vars, code_ppl, prob_ppl, curr_temp,
    ):
        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(
                    unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        return y, negs, num_vars, code_ppl, prob_ppl, curr_temp

    def apply_mask_quasiwave(self, x, padding_mask, mask_indices=None):
        B, T, C = x.shape
        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            else:
                if isinstance(mask_indices, np.ndarray):
                    mask_indices = torch.from_numpy(mask_indices)
                mask_indices = mask_indices.to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices


class TransformerQuasiwaveEncoder(TransformerEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.pre_transformer_ops = []

    def register_pre_transformer_op(self, op_fn):
        self.pre_transformer_ops.append(op_fn)

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        for _op_fn in self.pre_transformer_ops:
            x, padding_mask = _op_fn(x, padding_mask)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, padding_mask


@register_model_architecture("wav2vec2_quasiwave", "wav2vec2_quasiwave_base")
def base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.final_dim = getattr(args, "final_dim", 0)

    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    conv_feature_layers += " + [(512, 1, 1)]"
    args.conv_feature_layers = getattr(
        args, "conv_feature_layers", conv_feature_layers)

    args.logit_temp = getattr(args, "logit_temp", 0.1)

    args.quantize_targets = getattr(args, "quantize_targets", False)
    args.quantize_input = getattr(args, "quantize_input", False)
    args.same_quantizer = getattr(args, "same_quantizer", False)

    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)

    args.latent_vars = getattr(args, "latent_vars", 320)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.65)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)

    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0)
    args.mask_channel_selection = getattr(
        args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(
        args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.num_negatives = getattr(args, "num_negatives", 100)
    args.negatives_from_everywhere = getattr(
        args, "negatives_from_everywhere", False)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)
    args.codebook_negatives = getattr(args, "codebook_negatives", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)

    args.latent_temp = getattr(args, "latent_temp", "(2,0.5,0.999995)")

    args.target_glu = getattr(args, "target_glu", False)

    args.conv_bias = getattr(args, "conv_bias", False)
