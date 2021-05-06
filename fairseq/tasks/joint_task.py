# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json
from omegaconf import OmegaConf
from collections import OrderedDict
import sys

from fairseq.data.joint_dataset import JointDataset
from fairseq.models.joint_model import JointModel
from fairseq.criterions.joint_criterion import JointCriterion
from fairseq import tasks as fairseq_tasks
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq import metrics
from fairseq.joint_generator import JointGenerator
from fairseq.joint_utils import \
    JointTokenizer, JointBPE, JointScorer, \
    JointTargetDictionary, JointSourceDictionary, ShiftModelContext, \
    JointConfig
from fairseq import utils


logger = logging.getLogger(__name__)


@register_task("joint_task")
class JointTrainingTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--task-configs",
            type=str,
            nargs='+',
            help="Configuration YAML filename",
        )
        parser.add_argument(
            "--main-task-name",
            type=str,
            help="from which task the model is used"
        )
        parser.add_argument(
            "--sample-ratios",
            type=float,
            nargs='+',
            default=[1],
            help="sample ratio for each task"
        )
        parser.add_argument(
            "--loss-ratios",
            type=float,
            nargs='+',
            default=[1],
            help="sample ratio for each task"
        )

    def __init__(self, args, tasks, task_configs, main_task_name):
        super().__init__(args)
        self.task_names = args.task_configs
        self.tasks = tasks
        self.task_configs = task_configs
        self.main_task_name = main_task_name
        self.is_testing = 'generate' in sys.argv[0]

        if len(args.sample_ratios) != len(tasks):
            assert len(args.sample_ratios) == 1
            args.sample_ratios *= len(tasks)
        if len(args.loss_ratios) != len(tasks):
            assert len(args.loss_ratios) == 1
            args.loss_ratios *= len(tasks)
        self.sample_ratios = dict(zip(self.tasks.keys(), args.sample_ratios))
        self.loss_ratios = dict(zip(self.tasks.keys(), args.loss_ratios))
        self._tgt_dict = None
        self._src_dict = None
        self.cached_modules = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        tasks = OrderedDict()
        task_configs = OrderedDict()

        for config_file in args.task_configs:
            with open(config_file, 'r') as f:
                _task_config = json.load(f)
            _config = OmegaConf.create(_task_config)
            task_configs[config_file] = _config
            tasks[config_file] = fairseq_tasks.setup_task(_config.task)

        return cls(args, tasks, task_configs, args.main_task_name)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        split_codebook = {
            'train': 'train_subset',
            'valid': 'valid_subset', 'dev': 'valid_subset',
            'gen': 'gen_subset', 'test': 'gen_subset',
        }
        split_datasets = OrderedDict()
        for _name, _task in self.tasks.items():
            split_name = getattr(
                self.task_configs[_name].dataset, split_codebook[split], None)
            _task.load_dataset(split_name, epoch, combine, **kwargs)
            split_datasets[_name] = _task.datasets[split_name]
        self.datasets[split] = JointDataset(
            split_datasets,
            dataset_configs={
                _name: self.task_configs[_name].dataset
                for _name in self.tasks.keys()
            },
            sample_ratios=self.sample_ratios if split == 'train' else 1
        )

    def match_task_names(self, test_dict_to_match: dict):
        result = {}
        for _key, _value in test_dict_to_match.items():
            assert _key.startswith('test_') and _key[5:] in self.task_names
            result[_key[5:]] = _value
        return result

    def match_self_all(self):
        self.tasks = self.match_task_names(self.tasks)
        self.task_configs = self.match_task_names(self.task_configs)
        self.source_dictionary.dictionaries = self.match_task_names(
            self.source_dictionary.dictionaries
        )
        self.target_dictionary.dictionaries = self.match_task_names(
            self.target_dictionary.dictionaries
        )
        for _dataset in self.datasets.values():
            for _attr in ['datasets', 'sample_ratios', 'cumulative_sizes',
                          'cumulative_net_sizes', 'real_sizes',
                          'dataset_configs']:
                setattr(_dataset, _attr, self.match_task_names(
                    getattr(_dataset, _attr)
                ))

    def build_model(self, args):
        assert args._name == 'joint_model'
        if not self.is_testing:
            model = JointModel.build_model(
                self.task_configs, self.tasks, self.main_task_name
            )
        else:
            self.train_task = self.__class__.setup_task(args)
            self.task_names = list(self.train_task.tasks.keys())
            model = JointModel.build_model(
                self.train_task.task_configs, self.train_task.tasks,
                self.main_task_name
            )
            self.match_self_all()
        return model

    def build_criterion(self, args):
        assert args._name == 'joint_criterion'
        criterion = JointCriterion.build_criterion(
            self.task_configs, self.tasks, self.main_task_name
        )
        return criterion

    def build_generator(
            self, models, args,
            seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        generator = JointGenerator.build_generator(
            self.task_configs, self.tasks, self.main_task_name,
            models, seq_gen_cls, extra_gen_cls_kwargs,
        )
        return generator

    def build_tokenizer(self, tokenizer_config):
        self.tokenizer = JointTokenizer.build_tokenizer(
            self.task_configs
        )
        return self.tokenizer

    def build_bpe(self, bpe_config):
        self.bpe = JointBPE.build_bpe(
            self.task_configs, self.tasks
        )
        return self.bpe

    def post_process_cfg(self, cfg):
        self.cfg = JointConfig(cfg, self.task_configs)
        return self.cfg

    def shift_other_variables(self, cfg, *args):
        cfg = self.task_configs[self.main_task_name]
        align_dict = utils.load_align_dict(cfg.generation.replace_unk)
        src_dict = self.source_dictionary._src_dict
        tgt_dict = self.target_dictionary._tgt_dict
        return (
            cfg, align_dict, src_dict, tgt_dict
        )

    def build_scorer(self, scoring_config, tgt_dict):
        self.scorer = JointScorer.build_scorer(
            self.task_configs, tgt_dict
        )
        return self.scorer

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def src_dict(self):
        if self._src_dict is None:
            self._src_dict = JointSourceDictionary.build_dictionary(
                self.tasks, self.main_task_name
            )
        return self._src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def tgt_dict(self):
        if self._tgt_dict is None:
            self._tgt_dict = JointTargetDictionary.build_dictionary(
                self.tasks, self.main_task_name
            )
        return self._tgt_dict

    def criterion_wrapper(self, criterion, loss_ratio):

        def wrapper(*args, **kwargs):
            loss, sample_size, logging_output = \
                criterion(*args, **kwargs)
            weighted_loss = loss * loss_ratio
            logging_output['loss'] = logging_output['loss'] * loss_ratio
            return weighted_loss, sample_size, logging_output

        return wrapper

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        assert isinstance(dataset, JointDataset)
        return dataset.filter_indices_by_size(
            indices, max_positions, ignore_invalid_inputs
        )

    def shift_model(self, task_name, model):
        raise NotImplementedError()

    def shift_model_back(self, task_name, model):
        raise NotImplementedError()

    def shift_generator(self, task_name, generator):
        generator.main_task_name_cache = generator.main_task_name
        generator.main_task_name = task_name
        self.main_task_name = task_name
        self.tokenizer.main_task_name = task_name
        self.bpe.main_task_name = task_name
        self.scorer.main_task_name = task_name
        self.tgt_dict.main_task_name = task_name
        self.src_dict.main_task_name = task_name

    '''
    def shift_generator_back(self, generator):
        generator.main_task_name = generator.main_task_name_cache
        generator.main_task_name_cache = None
        self.main_task_name = None
        self.tokenizer.main_task_name = None
        self.bpe.main_task_name = None
        self.scorer.main_task_name = None
        self.tgt_dict.main_task_name = None
        self.src_dict.main_task_name = None
    '''

    def any_step(self, step_type: str, sample, model, criterion,
                 *args, **kwargs):
        assert isinstance(criterion, JointCriterion)
        assert isinstance(model, JointModel) or \
            '_DistributedFairseqModel' in str(model.__class__)
        task_name = sample['task_name']
        task = self.tasks[task_name]
        local_criterion = self.criterion_wrapper(
            criterion[task_name],
            self.loss_ratios[task_name],
        )
        local_model = model.main_model

        if step_type == 'train':
            step_fn = task.train_step
        elif step_type == 'valid':
            step_fn = task.valid_step
        else:
            raise Exception(f"step type {step_type} not recognized")

        with ShiftModelContext(self, task_name, model):
            loss, sample_size, logging_output = step_fn(
                    sample, local_model, local_criterion,
                    *args, **kwargs
                )

        logging_output['task_name'] = task_name
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion,
                   optimizer, update_num, ignore_grad):
        return self.any_step(
            'train',
            sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        return self.any_step(
            'valid',
            sample, model, criterion
        )

    def inference_step(self, generator, models, sample,
                       prefix_tokens=None, constraints=None):
        assert len(models) == 1
        task_name = sample['task_name']
        self.shift_generator(task_name, generator)
        with ShiftModelContext(self, task_name, models[0]):
            hypos = generator.generate(
                task_name, models, sample,
                prefix_tokens=prefix_tokens, constraints=constraints
            )
        return hypos

    def cache_module(self, cache_name, module_name, target_model, from_model):
        self.cached_modules[cache_name] = \
            getattr(target_model, module_name)
        setattr(
            target_model,
            module_name,
            getattr(from_model, module_name)
        )

    def cache_module_recover(self, cache_name, module_name, target_model):
        setattr(
            target_model,
            module_name,
            self.cached_modules.pop(cache_name),
        )

    def reduce_metrics(self, logging_outputs, criterion):
        if metrics._active_aggregators_cnt.get('train', 0) != 0:
            status = 'train'
        else:
            status = 'valid'

        for _name, _task in self.tasks.items():
            local_criterion = criterion.criterions[_name]
            local_loggings = [_log for _log in logging_outputs
                              if _log['task_name'] == _name]
            if len(local_loggings) > 0:
                if status == 'train':
                    with metrics.aggregate(':'.join(('train', _name))):
                        with metrics.aggregate(
                                ':'.join(('train_inner', _name))):
                            _task.reduce_metrics(
                                local_loggings, local_criterion)
                else:
                    with metrics.aggregate(':'.join(('valid', _name))):
                        _task.reduce_metrics(local_loggings, local_criterion)

    def log_private_metrics(
            self, log_fn, log_type, step, post_processing=None):
        for _name in self.task_names:
            metrics_name = ':'.join((log_type, _name))
            if metrics_name not in metrics._aggregators:
                continue
            stats = metrics.get_smoothed_values(metrics_name)
            if post_processing is not None:
                stats = post_processing(stats)
            log_fn(stats, tag=metrics_name, step=step)
            metrics.reset_meters(_name)
