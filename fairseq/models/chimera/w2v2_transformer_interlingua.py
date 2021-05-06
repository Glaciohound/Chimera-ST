#!/usr/bin/env python3

import logging

import torch
import torch.nn as nn
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import Embedding
from fairseq.modules import (
    TransformerEncoderLayer,
)

from .w2v2_transformer import \
    base_architecture, S2TTransformerModelW2V2, S2T_W2V2_TransformerEncoder
from fairseq.models.speech_to_text.s2t_transformer import \
    TransformerDecoderScriptable
from .chi_utils import update_dict_with_prefix


logger = logging.getLogger(__name__)


@register_model("s2t_transformer_w2v2_interlingua")
class S2TTransformerInterlinguaModelW2V2(S2TTransformerModelW2V2):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder, debug_options):
        super().__init__(encoder, decoder)
        self.debug_options = debug_options

    @staticmethod
    def add_args(parser):
        S2TTransformerModelW2V2.add_args(parser)
        parser.add_argument(
            '--interlingua-length', type=int, default=16
        )
        parser.add_argument(
            '--interlingua-layers', type=int, default=3
        )
        parser.add_argument(
            '--interlingua-debug-options',
            type=str,
            nargs='+',
            default=[],
            choices=['modal_embedding'],
            help='extra options for debug use only'
        )
        parser.add_argument(
            '--non-shared-encoder-layers',
            type=int,
            default=0,
        )

        parser.add_argument('--fix-wav2vec', action='store_true',
                            default=False)
        parser.add_argument('--fix-interlingua', action='store_true',
                            default=False)
        parser.add_argument('--fix-decoder', action='store_true',
                            default=False)
        parser.add_argument('--fix-decoder-transformers', action='store_true',
                            default=False)
        parser.add_argument('--fix-encoder-transformers', action='store_true',
                            default=False)
        parser.add_argument('--reset-encoder', action='store_true',
                            default=False)
        parser.add_argument('--no-interlingua', action='store_true',
                            default=False)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        s2t_transformer_w2v2_interlingua_base(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        if task.source_dictionary is not None:
            encoder_embed_tokens = build_embedding(
                task.source_dictionary, args.encoder_embed_dim
            )
        else:
            encoder_embed_tokens = None
        encoder = cls.build_encoder(
            args, task.source_dictionary,
            encoder_embed_tokens)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        decoder = cls.build_decoder(
            args, task.target_dictionary, decoder_embed_tokens)

        if args.fix_wav2vec:
            logger.info('setting all parameters in wav2vec_model to fixed')
            encoder.wav2vec_model.requires_grad_(False)
        if args.fix_encoder_transformers:
            logger.info('setting transformer layers in decoder to fixed')
            encoder.transformer_layers.requires_grad_(False)
        if args.fix_decoder_transformers:
            logger.info('setting transformer layers in decoder to fixed')
            decoder.layers.requires_grad_(False)
        if args.fix_decoder:
            logger.info('setting all parameters in decoder to fixed')
            decoder.requires_grad_(False)
        if args.fix_interlingua:
            logger.info('setting all parameters in interlingua to fixed')
            encoder.interlingua_layers.requires_grad_(False)
            encoder.interlingua_embedding.requires_grad_(False)

        return cls(encoder, decoder, args.interlingua_debug_options)

    @classmethod
    def build_encoder(cls, args, src_dict=None, encoder_embed_tokens=None):
        encoder = S2T_W2V2_TransformerInterlinguaEncoder(
            args, src_dict, encoder_embed_tokens,
        )
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderScriptable(
            args, tgt_dict, embed_tokens)

    def forward_with_internal(
        self, src_tokens, src_lengths,
        prev_output_tokens, **extra_args
    ):
        encoder_out = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out, encoder_out.encoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        state_dict.pop("encoder.stashed_weights", None)
        state_dict.pop("decoder.stashed_weights", None)
        return state_dict


class S2T_W2V2_TransformerInterlinguaEncoder(S2T_W2V2_TransformerEncoder):
    """Speech-to-text Transformer encoder that consists of
    input wav2vec2Encoder, subsampler and
    Transformer encoder."""

    def __init__(self, args, src_tokens, embed_tokens):
        super().__init__(args)

        self.max_source_positions = args.max_source_positions
        self.text_embed_tokens = embed_tokens
        if embed_tokens is not None:
            self.encoder_embed_dim = embed_tokens.embedding_dim
        self.debug_options = args.interlingua_debug_options
        self.non_shared_encoder_layers = args.non_shared_encoder_layers
        self.reset_encoder = args.reset_encoder
        self.no_interlingua = args.no_interlingua

        assert args.interlingua_layers >= 1
        self.interlingua_embedding = Embedding(
            args.interlingua_length, args.encoder_embed_dim, 0)
        self.interlingua_layers = nn.ModuleList(
            [TransformerEncoderLayer(args)
             for _ in range(args.interlingua_layers)]
        )
        if 'modal_embedding' in self.debug_options:
            self.modal_embedding = Embedding(3, args.encoder_embed_dim, 2)
        else:
            self.modal_embedding = None

        if self.non_shared_encoder_layers > 0:
            self.audio_exclusive_layers = nn.ModuleList(
                [TransformerEncoderLayer(args)
                 for _ in range(self.non_shared_encoder_layers)]
            )

    def upgrade_state_dict_named(self, state_dict, name):
        if self.reset_encoder:
            logger.info("resetting encoder's transformer weights")
            for module_name in ("embed_positions", "transformer_layers"):
                module = getattr(self, module_name)
                update_dict_with_prefix(
                    module.state_dict(), state_dict, f"{name}.{module_name}.")

        embed_weight_name = name+'.text_embed_tokens.weight'
        if self.text_embed_tokens is None:
            logger.info(f"removing unused weight {embed_weight_name}")
            state_dict.pop(embed_weight_name)
        return state_dict

    def max_positions(self):
        return None

    def forward(self, src_tokens, src_lengths, **extra_args):
        """
        :param src_tokens: b x frames
        :param src_lengths: b-dim
        """
        is_text = not src_tokens.dtype.is_floating_point
        if is_text:
            # text input
            # embedding
            feature = self.text_embed_tokens(src_tokens).transpose(0, 1)
            input_lengths = src_lengths
        else:
            input_tokens, input_lengths = src_tokens, src_lengths

            # 1. wav2vec
            # print(src_tokens.size(), src_lengths.size())
            feature, _, input_lengths = \
                self._get_w2v_feature(input_tokens, input_lengths)

            # 2. conv-layers
            # print("after w2v extract, x:", w2v_feature.size())
            feature, input_lengths = self.subsample(feature, input_lengths)

        # x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * feature
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        if is_text:
            positions = self.embed_positions(
                encoder_padding_mask).transpose(0, 1)
            x += positions
        x = self.dropout_module(x)

        # 3. Transformer-layers
        if is_text or self.non_shared_encoder_layers == 0:
            for layer in self.transformer_layers:
                x = layer(x, encoder_padding_mask)
        else:
            for layer in self.audio_exclusive_layers:
                x = layer(x, encoder_padding_mask)
            for layer in self.transformer_layers[
                self.non_shared_encoder_layers:
            ]:
                x = layer(x, encoder_padding_mask)

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        length_h, batch_size, _ = x.shape
        device = x.device

        if self.no_interlingua:
            interlingua = x
            length_i = interlingua.shape[0]
        else:
            # 4. Interlingua representation
            h_enc = x
            device = h_enc.device

            interlingua = self.interlingua_embedding.weight\
                .unsqueeze(1).repeat(1, batch_size, 1)
            length_i = interlingua.shape[0]

            if self.modal_embedding is not None:
                if is_text:
                    interlingua += self.modal_embedding(
                        torch.ones(length_i, batch_size,
                                   device=device, dtype=int)
                    )
                else:
                    interlingua += self.modal_embedding(
                        torch.zeros(length_i, batch_size,
                                    device=device, dtype=int)
                    )

            attn_mask = torch.ones(
                (length_h+length_i, length_h+length_i),
                dtype=float, device=device,
            )
            attn_mask[:, :length_h] = 0
            for _i, _layer in enumerate(self.interlingua_layers):
                interlingua = _layer(
                    x=torch.cat((h_enc, interlingua), 0),
                    encoder_padding_mask=torch.zeros(
                        batch_size, length_h + length_i,
                        dtype=bool, device=device
                    ),
                    attn_mask=attn_mask
                )
                interlingua = interlingua[-length_i:]

        # final: prepare output
        encoder_padding_mask = torch.zeros(
            batch_size, length_i,
            device=device, dtype=bool
        )
        return EncoderOut(
            encoder_out=interlingua,
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=None,
        )


@register_model_architecture(
    "s2t_transformer_w2v2_interlingua",
    "s2t_transformer_w2v2_interlingua_base")
def s2t_transformer_w2v2_interlingua_base(args):
    base_architecture(args)
    args.use_asr_finetune_w2v = getattr(args, "use_asr_finetune_w2v", False)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", 256 * 8)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

    args.max_source_positions = getattr(args, 'max_source_positions', 1000000)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.dropout = getattr(args, "dropout", 0.1)

    args.fix_wav2vec = getattr(args, 'fix_wav2vec', False)
    args.load_pretrained_encoder_from = getattr(
        args, 'load_pretrained_encoder_from', None)
    args.non_shared_encoder_layers = getattr(
        args, "non_shared_encoder_layers", 0)
    args.fix_wav2vec = getattr(args, 'fix_wav2vec', False)
    args.fix_encoder_transformers = getattr(
        args, 'fix_decoder_transformers', False)
    args.fix_decoder_transformers = getattr(
        args, 'fix_decoder_transformers', False)
    args.fix_decoder = getattr(args, 'fix_decoder', False)
    args.fix_interlingua = getattr(args, 'fix_interlingua', False)
    args.no_interlingua = getattr(args, 'no_interlingua', False)
    args.reset_encoder = getattr(args, 'reset_encoder', False)
