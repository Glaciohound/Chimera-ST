# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import logging
from typing import Union
import random

import numpy as np
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict, defaultdict
from fairseq import utils

from . import FairseqDataset

logger = logging.getLogger(__name__)


class JointDataset(FairseqDataset):
    @staticmethod
    def cumsum(datasets, sample_ratios):
        r, s = OrderedDict(), 0
        for _name in datasets.keys():
            curr_len = int(sample_ratios[_name] * len(datasets[_name]))
            r[_name] = curr_len + s
            s += curr_len
        return r

    def __init__(self, datasets: dict, dataset_configs: dict,
                 sample_ratios: Union[int, dict]):
        super(JointDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = datasets
        if isinstance(sample_ratios, int):
            sample_ratios = {
                _name: sample_ratios for _name in self.datasets.keys()
            }
        self.sample_ratios = sample_ratios
        self.cumulative_sizes = self.cumsum(self.datasets, sample_ratios)
        self.cumulative_net_sizes = self.cumsum(
            self.datasets, defaultdict(lambda: 1))
        self.real_sizes = {
            _name: len(_dataset) for _name, _dataset in self.datasets.items()
        }
        self.dataset_configs = dataset_configs
        logger.info(self.__repr__())

    def __len__(self):
        return list(self.cumulative_sizes.values)[-1]

    def __getitem__(self, idx):
        task_name, sample_idx = self._get_dataset_and_sample_index(idx)
        sample = {
            'task_name': task_name,
            'data': self.datasets[task_name][sample_idx]
        }
        return sample

    @staticmethod
    def bisect_dict(to_bisect: OrderedDict, idx: Union[int, list]):
        keys = list(to_bisect.keys())
        values = list(to_bisect.values())
        if isinstance(idx, int):
            key_idx = bisect.bisect_right(values, idx)
            out_key = keys[key_idx]
        else:
            key_idx = [bisect.bisect_right(values, _i) for _i in idx]
            out_key = [keys[_i] for _i in key_idx]
        return key_idx, out_key

    def indices_belongs_to(self, indices):
        dataset_indices, dataset_names = self.bisect_dict(
            self.cumulative_net_sizes, indices
        )
        return dataset_indices, dataset_names

    def _get_dataset_and_sample_index(self, idx: int):
        _, task_name = \
            self.bisect_dict(self.cumulative_net_sizes, idx)
        sample_idx = self.convert_indices([idx], task_name, True)[0]
        # sample_idx = sample_idx % self.real_sizes[dataset_name]
        return task_name, sample_idx

    def collater(self, samples, **extra_args):
        if len(samples) > 0:
            dataset_names = list(sample['task_name'] for sample in samples)
            dataset_name = random.choice(dataset_names)
            count = dataset_names.count(dataset_name)
            if count != len(dataset_names):
                raise Exception("encountering a mixed sample set")
            to_collate = [sample['data'] for sample in samples]
            if hasattr(self.datasets[dataset_name], 'collater'):
                collated = self.datasets[dataset_name].collater(
                    to_collate, **extra_args)
            else:
                collated = default_collate(samples, **extra_args)
        else:
            return None

        collated['task_name'] = dataset_name
        return collated

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        task_name, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[task_name].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        _, dataset_name = self.bisect_dict(
            self.cumulative_net_sizes, index)
        return getattr(self.datasets[dataset_name], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = OrderedDict()
        for _name, ds in self.datasets.items():
            sizes = ds.sizes
            if isinstance(sizes, list):
                sizes = sizes[0]
            assert isinstance(sizes, np.ndarray)
            if sizes.ndim > 1:
                assert sizes.ndim == 2
                sizes = sizes[:, 0]
            _dataset_sizes[_name] = sizes
        return np.concatenate(list(_dataset_sizes.values()))

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets.values())

    def ordered_indices(self):
        """
        Returns indices sorted by length. So less padding is needed.
        """
        individual_samples = []
        for _name, _dataset in self.datasets.items():
            ordered = _dataset.ordered_indices()
            # the sample ratio may be float
            sr = self.sample_ratios[_name]
            round_int, residue = int(sr), sr - int(sr)
            sample_idx = np.sort(np.random.randint(
                len(ordered),
                size=int(residue * len(_dataset))))
            sampled = np.concatenate((
                np.tile(ordered, round_int),
                ordered[sample_idx]
            ))
            logger.info(f"sampled {len(sampled)} indices from {_name} dataset")
            sampled = self.convert_indices(
                sampled, _name, out_to_in=False)
            individual_samples.append(sampled)
        samples = np.concatenate(individual_samples)
        return samples

    def prefetch(self, indices):
        frm = 0
        for _name, ds in self.datasets.items():
            to = self.cumulative_net_sizes[_name]
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size
                             for i in indices if frm <= i < to])
            frm = to

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(d.can_reuse_epoch_itr_across_epochs
                   for d in self.datasets.vavalues())

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets.values():
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def convert_indices(self, indices, dataset_name, out_to_in: bool):
        """ converting between dataset-local indices and inner-indices """
        is_list = isinstance(indices, list)
        if is_list:
            indices = np.array(indices)
        offset = self.cumulative_net_sizes[dataset_name] - \
            len(self.datasets[dataset_name])
        if out_to_in:
            indices -= offset
        else:
            indices += offset
        return indices.tolist() if is_list else indices

    def filter_indices_by_size(
        self, indices, max_positions=None, ignore_invalid_inputs=False
    ):
        _, dataset_names = self.indices_belongs_to(indices)
        dataset_names = np.array(dataset_names)
        valid_flag = np.array(True).repeat(len(indices))
        belonging_mask = {
            _name: _name == dataset_names
            for _name in self.datasets.keys()
        }
        indices_array = np.array(indices)

        for _name, _dataset in self.datasets.items():
            task_max_positions = utils.resolve_max_positions(
                max_positions, self.tasks[_name].max_positions()
            )
            logger.info(
                f"filtering dataset {_name}")

            mask = belonging_mask[_name]
            valid, ignored = _dataset.filter_indices_by_size(
                self.convert_indices(
                    indices[mask],
                    _name, out_to_in=True
                ),
                task_max_positions
            )
            if len(ignored) > 0:
                logger.warning(
                    (
                        "{} samples have invalid sizes and will be skipped, "
                        "max_positions={}, first few sample ids={}"
                    ).format(len(ignored), task_max_positions, ignored[:10])
                )
                if not ignore_invalid_inputs:
                    raise Exception("encountered with invalid inputs")

            local_valid_flag = valid_flag[mask].copy()
            local_valid_flag[ignored] = False
            valid_flag[mask] = local_valid_flag

        indices_array = indices_array[valid_flag]
        return indices_array.tolist()

    def batch_by_size(
        self, indices, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1,
    ):
        _, dataset_names = self.bisect_dict(
            self.cumulative_net_sizes, indices)
        dataset_names = np.array(dataset_names)
        indices_array = np.array(indices)
        batch_groups = []

        for _name, _dataset in self.datasets.items():
            local_indices = self.convert_indices(
                indices_array[dataset_names == _name],
                _name, out_to_in=True
            )
            local_max_tokens = self.dataset_configs[_name].max_tokens
            if max_tokens is not None and max_tokens > 0:
                local_max_tokens = min(max_tokens, local_max_tokens)

            local_batches = _dataset.batch_by_size(
                local_indices.tolist(),
                local_max_tokens, max_sentences,
                required_batch_size_multiple
            )
            local_batches = [
                self.convert_indices(_batch, _name, out_to_in=False)
                for _batch in local_batches
            ]
            logger.info(f"grouped {len(local_batches)} batches from {_name}")
            batch_groups += local_batches

        return batch_groups
