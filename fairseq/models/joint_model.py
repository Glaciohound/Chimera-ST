#!/usr/bin/env python3

# from argparse import Namespace
import logging

# import torch
# import torch.nn as nn
# from fairseq import checkpoint_utils, utils, tasks
# from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)


logger = logging.getLogger(__name__)


@register_model("joint_model")
class JointModel(BaseFairseqModel):
    """
    A specially designed joint model for joint training.
    It nearly does nothing, except distributing workloads to its sub-models.
    """

    def __init__(self, models: dict, task_configs: dict, main_task_name: str):
        super().__init__()
        self.models = models
        self.main_task_name = main_task_name
        self._initial_main_task_name = main_task_name
        self.task_configs = task_configs
        for _name, _model in self.models.items():
            _name = _name.replace('.', '_')
            setattr(self, _name, _model)
            self.add_module(_name, _model)

    @property
    def main_model(self):
        return self.models[self._initial_main_task_name]

    '''
    def state_dict(self):
        return {
            _name: _model.state_dict()
            for _name, _model in self.models.items()
        }

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        for _name, _model in self.models.items():
            _model.load_state_dict(
                state_dict[_name], strict=strict,
                model_cfg=self.task_configs[_name].model,
            )

    def upgrade_state_dict(self, state_dict, name):
        for _name, _model in self.models.items():
            _model.upgrade_state_dict(state_dict[_name])
    '''

    @classmethod
    def build_model(cls, task_configs, tasks, main_task_name):
        models = {}
        for _name, _task in tasks.items():
            models[_name] = _task.build_model(
                task_configs[_name].model
            )
            logger.info(f"bulit model {models[_name].__class__}"
                        f" for {_name} {_task}")
        return cls(models, task_configs, main_task_name)

    def forward(self, *args, **kwargs):
        return self.main_model(*args, **kwargs)

    def __getitem__(self, task_name):
        return self.models[task_name]

    def to(self, *args, **kwargs):
        for _model in self.models.values():
            _model.to(*args, **kwargs)
        return self

    def cpu(self):
        for _model in self.models.values():
            _model.cpu()
        return self

    def cuda(self):
        for _model in self.models.values():
            _model.cuda()
        return self

    def half(self):
        for _model in self.models.values():
            _model.half()
        return self

    def float(self):
        for _model in self.models.values():
            _model.float()
        return self

    def train(self, *args, **kwargs):
        for _model in self.models.values():
            _model.train(*args, **kwargs)
        return self

    def eval(self, *args, **kwargs):
        for _model in self.models.values():
            _model.eval()
        return self

    def max_decoder_positions(self):
        return min([m.decoder.max_positions() for m in self.models.values()])

    @property
    def encoder(self):
        return self.main_model.encoder

    @property
    def decoder(self):
        return self.main_model.decoder

    def named_parameters(self, recurse: bool = True):
        for _name, _model in self.models.items():
            for _pname, _param in _model.named_parameters():
                yield _name+':'+_pname, _param


@register_model_architecture(model_name="joint_model",
                             arch_name="joint_model")
def base_architecture(args):
    pass
