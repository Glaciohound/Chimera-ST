import torch
import torch.nn as nn
import numpy as np


class LengthDropout(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self._dropout = dropout

    @property
    def dropout(self):
        if self.training:
            return self._dropout
        else:
            return 0

    def forward(self, x, axis=0, along_dims=()):
        if self.dropout == 0:
            return x, None

        assert axis not in along_dims
        along_dims = sorted(along_dims)

        length = x.shape[axis]
        num_selected = int(length * (1 - self.dropout))

        along_dims_shape = [x.shape[_i] for _i in along_dims]
        if len(along_dims) == 0:
            along_dims_numel = 1
        else:
            along_dims_numel = np.prod(along_dims_shape)
        along_ndim = len(along_dims)
        extra_dims_shape = [x.shape[_i] for _i in range(x.ndim)
                            if _i not in along_dims and _i != axis]
        extra_ndim = len(extra_dims_shape)

        vital_indices = torch.tensor([
            np.sort(np.random.choice(
                range(length), num_selected, replace=False
            ))
            for _i in range(along_dims_numel)
        ]).to(x.device)
        indices = vital_indices.view(
            along_dims_shape + [-1] + [1] * extra_ndim
        ).repeat([1] * (along_ndim+1) + extra_dims_shape)

        permute_shape = [None] * x.ndim
        for _i, _d in enumerate(along_dims):
            permute_shape[_d] = _i
        permute_shape[axis] = along_ndim
        _count = along_ndim
        for _i, _d in enumerate(permute_shape):
            if _d is None:
                _count += 1
                permute_shape[_i] = _count

        indices = indices.permute(permute_shape)
        selected = x.gather(axis, indices)
        return selected, indices


def search_and_replace_within_self_and_cached(
    model, name, module, state_dict, logger
):
    logger.info(f"current {name} replaced into state_dict")
    state_dict[name] = module.weight
    for i, cand in enumerate(model.stashed_weights.get(
         name, []
    )):
        if cand.shape == module.weight.shape:
            logger.info(f"alternatives for {name} found in cached, "
                        f"index {i}")
            state_dict[name] = cand


def update_dict_with_prefix(from_dict: dict, to_dict: dict, prefix):
    from_dict_prefixed = {
        prefix + _name: _value for _name, _value in from_dict.items()
    }
    to_dict.update(from_dict_prefixed)
