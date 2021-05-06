# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class JointGenerator(nn.Module):
    def __init__(
        self, generators: dict, main_task_name: str
    ):
        super().__init__()
        self.generators = generators
        self.main_task_name = main_task_name
        self.main_task_name_cache = None

    @classmethod
    def build_generator(
        cls,
        task_configs, tasks, main_task_name,
        models, seq_gen_cls=None, extra_gen_cls_kwargs=None,
    ):
        generators = {}
        for _name, _task in tasks.items():
            generators[_name] = _task.build_generator(
                models, task_configs[_name].generation,
                seq_gen_cls=seq_gen_cls,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )
            logger.info(f"bulit generator {generators[_name].__class__}"
                        f" for {_name} {_task}")
        return cls(generators, main_task_name)

    def cuda(self):
        for _generator in self.generators.values():
            _generator.cuda()
        return self

    @torch.no_grad()
    def forward(
        self, *args, **kwargs
    ):
        raise NotImplementedError()

    @torch.no_grad()
    def generate(self, task_name: str, *args, **kwargs):
        local_generator = self.generators[task_name]
        return local_generator.generate(*args, **kwargs)

    @property
    def main_generator(self):
        return self.generators[self.main_task_name]

    @property
    def symbols_to_strip_from_output(self):
        return self.main_generator.symbols_to_strip_from_output

    @property
    def eos(self):
        return self.main_generator.eos
