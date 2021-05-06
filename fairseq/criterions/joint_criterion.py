# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.criterions import FairseqCriterion, register_criterion

logger = logging.getLogger(__name__)


@register_criterion("joint_criterion")
class JointCriterion(FairseqCriterion):
    def __init__(self, criterions: dict, main_task_name: str):
        super().__init__(criterions[main_task_name])
        self.criterions = criterions
        self.main_task_name = main_task_name
        self.main_criterion = criterions[main_task_name]

    @classmethod
    def build_criterion(cls, task_configs, tasks, main_task_name):
        criterions = {}
        for _name, _task in tasks.items():
            criterions[_name] = _task.build_criterion(
                task_configs[_name].criterion
            )
            logger.info(f"bulit criterion {criterions[_name].__class__}"
                        f" for {_name} {_task}")
        return cls(criterions, main_task_name)

    def forward(self, model, sample, reduce=True):
        return NotImplementedError()

    def __getitem__(self, task_name):
        return self.criterions[task_name]

    def get_lprobs_and_target(self, model, net_output, sample):
        raise NotImplementedError()

    def compute_loss(self, model, net_output, sample, reduce=True):
        raise NotImplementedError()

    def compute_accuracy(self, model, net_output, sample):
        raise NotImplementedError()

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        raise NotImplementedError()

    def train(self, *args, **kwargs):
        for _criterion in self.criterions.values():
            _criterion.train(*args, **kwargs)
        return self

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
