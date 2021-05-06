# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.tasks import register_task
from fairseq.tasks.joint_task import JointTrainingTask


logger = logging.getLogger(__name__)


@register_task("joint_task_mtst")
class JointTrainingMTSTTask(JointTrainingTask):

    st_task = 'task_st.yaml'
    mt_task = 'task_mt.yaml'
    main_task = mt_task

    @staticmethod
    def add_args(parser):
        JointTrainingTask.add_args(parser)
        parser.add_argument(
            '--other-unshare-modules',
            nargs='+', type=str,
            choices=['encoder-transformers', 'decoder-transformers',
                     'interlingua-transformers'],
            default=[],
        )

    def __init__(self, args, tasks, task_configs, main_task_name):
        super().__init__(args, tasks, task_configs, main_task_name)
        self.other_unshare_modules = args.other_unshare_modules
        self.state = self.main_task

    def resetup(self, args):
        self.other_unshare_modules = args.other_unshare_modules

    def shift_model(self, task_name, model):
        st_model = model[self.st_task]
        mt_model = model[self.mt_task]
        if task_name == self.st_task:
            self.cache_module(
                'mt_decoder_embedding', 'embed_tokens',
                mt_model.decoder, st_model.decoder
            )
            self.cache_module(
                'mt_decoder_projection', 'output_projection',
                mt_model.decoder, st_model.decoder
            )
            if 'encoder-transformers' in self.other_unshare_modules:
                self.cache_module(
                    'mt_encoder_transformers', 'transformer_layers',
                    mt_model.encoder, st_model.encoder
                )
            if 'decoder-transformers' in self.other_unshare_modules:
                self.cache_module(
                    'mt_decoder_transformers', 'layers',
                    mt_model.decoder, st_model.decoder
                )
            if 'interlingua-transformers' in self.other_unshare_modules:
                self.cache_module(
                    'mt_interlingua_transformers', 'interlingua_layers',
                    mt_model.encoder, st_model.encoder
                )
            self.state = self.st_task
        elif task_name == self.mt_task:
            pass
        else:
            raise Exception()

    def shift_model_back(self, task_name, model):
        mt_model = model[self.mt_task]
        if task_name == self.st_task:
            self.cache_module_recover(
                'mt_decoder_embedding', 'embed_tokens',
                mt_model.decoder
            )
            self.cache_module_recover(
                'mt_decoder_projection', 'output_projection',
                mt_model.decoder
            )
            if 'encoder-transformers' in self.other_unshare_modules:
                self.cache_module_recover(
                    'mt_encoder_transformers', 'transformer_layers',
                    mt_model.encoder,
                )
            if 'decoder-transformers' in self.other_unshare_modules:
                self.cache_module_recover(
                    'mt_decoder_transformers', 'layers',
                    mt_model.decoder
                )
            if 'interlingua-transformers' in self.other_unshare_modules:
                self.cache_module_recover(
                    'mt_interlingua_transformers', 'interlingua_layers',
                    mt_model.encoder
                )
            self.state = self.mt_task
        elif task_name == self.mt_task:
            pass
        else:
            raise Exception()
