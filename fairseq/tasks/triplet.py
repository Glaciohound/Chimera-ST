# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace
import torch
import pickle
import numpy as np

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.triplet_dataset import (
    TripletDataConfig,
    TripletDataset,
    TripletDatasetCreator,
    get_features_or_waveform,
)
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset


logger = logging.getLogger(__name__)


@register_task("triplet")
class TripletTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--dump-feature-to-file",
            type=str, default=None,
        )
        parser.add_argument(
            "--sample-rate", type=int, default=16000
        )

    def __init__(self, args, tgt_dict, src_dict, data_cfg):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.src_dict = src_dict
        self.data_cfg = data_cfg
        self.dump_feature_to_file = args.dump_feature_to_file
        if self.dump_feature_to_file is not None:
            self.cached_features = {
                _name: [] for _name in
                ('src_text', 'audio_features', 'text_features')
            }
        else:
            self.cached_features = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = TripletDataConfig(op.join(args.data, args.config_yaml))

        def load_dict(vocab_filename):
            _dict_path = op.join(args.data, vocab_filename)
            if not op.isfile(_dict_path):
                raise FileNotFoundError(f"Dict not found: {_dict_path}")
            _dict = Dictionary.load(_dict_path)
            return _dict

        tgt_dict = load_dict(data_cfg.vocab_filename)
        src_dict = load_dict(data_cfg.src_vocab_filename)
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): "
            f"{len(tgt_dict):,}"
        )
        logger.info(
            f"source dictionary size ({data_cfg.src_vocab_filename}): "
            f"{len(src_dict):,}"
        )

        return cls(args, tgt_dict, src_dict, data_cfg)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_bpe_tokenizer = self.build_src_bpe()
        self.datasets[split] = TripletDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            normalize=self.args.normalize,
            sample_rate=self.args.sample_rate,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super().build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        self.tokenizer = encoders.build_tokenizer(
            Namespace(**self.data_cfg.pre_tokenizer))
        return self.tokenizer

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        self.bpe = encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))
        return self.bpe

    def build_src_bpe(self):
        logger.info(f"source tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        self.src_bpe = encoders.build_bpe(
            Namespace(**self.data_cfg.src_bpe_tokenizer))
        return self.src_bpe

    '''
    @classmethod
    def build_dataset_for_inference(cls, audio_paths, n_frames, **kwargs):
        return TripletDataset("interactive", False, {}, audio_paths, n_frames)
    '''

    def valid_step(self, sample, model, criterion):
        if self.dump_feature_to_file is not None:
            model.eval()
            with torch.no_grad():
                st_input = sample['net_input']
                mt_input = {
                    "src_tokens": sample["src_text"],
                    "src_lengths": sample["src_text_lengths"],
                    "prev_output_tokens":
                    sample["net_input"]["prev_output_tokens"],
                    "mask": sample["net_input"]["mask"],
                }
                _, audio_internal = model.forward_with_internal(**st_input)
                _, text_internal = model.forward_with_internal(**mt_input)
                self.cached_features['audio_features'].append(
                    audio_internal.detach().cpu().numpy().transpose(1, 0, 2),
                )
                self.cached_features['text_features'].append(
                    text_internal.detach().cpu().numpy().transpose(1, 0, 2),
                )
                self.cached_features['src_text'].extend([
                    self.datasets['dev_wave'].datasets[0].src_texts[i]
                    for i in sample['id']
                ])
        return super().valid_step(sample, model, criterion)

    def dump_features(self):
        if self.cached_features is None:
            return
        with open(self.dump_feature_to_file, 'wb') as f:
            self.cached_features['audio_features'] = np.concatenate(
                self.cached_features['audio_features']
            )
            self.cached_features['text_features'] = np.concatenate(
                self.cached_features['text_features']
            )
            pickle.dump(self.cached_features, f)

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p, True).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return TripletDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
