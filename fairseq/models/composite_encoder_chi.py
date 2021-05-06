# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .fairseq_encoder import FairseqEncoder
import torch
import contextlib


class CompositeEncoderChi(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    We run forward on each encoder and return a dictionary of outputs. The first
    encoder's dictionary is used for initialization.

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
    """

    def __init__(self, encoders):
        encoders = {name: enc for name, enc in encoders.items()
                    if enc is not None}
        super().__init__(next(iter(encoders.values())).dictionary)
        self.encoders = encoders
        for key in self.encoders:
            self.add_module(key, self.encoders[key])

    def forward(self, to_fix=[], **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`

        Returns:
            dict:
                the outputs from each Encoder
        """
        x = {'encoder_out': kwargs.pop('src_tokens')}
        for name, encoder in self.encoders.items():
            with torch.no_grad() if name in to_fix else contextlib.ExitStack():
                if encoder._get_name() == 'Wav2VecEncoder':
                    x = encoder(source=x['encoder_out'],
                                padding_mask=None, tbc=False, **kwargs)
                elif encoder._get_name() == 'S2TTransformerEncoder':
                    encoder_out = x['encoder_out']
                    x = encoder(
                        src_tokens=x['encoder_out'],
                        src_lengths=torch.tensor([
                            piece.shape[0] for piece in encoder_out])
                        .to(encoder_out.device)
                    )
                else:
                    raise NotImplementedError()
        return x

    def __getitem__(self, index):
        return self.encoders[index]

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to new_order."""
        encoder_out = list(self.encoders.values())[-1].reorder_encoder_out(
            encoder_out, new_order
        )
        return encoder_out

    def max_positions(self):
        each = [encoder.max_positions() for encoder in self.encoders.values()]
        each = [_num for _num in each if _num is not None]
        if len(each) > 0:
            return min(each)
        else:
            return None

    def upgrade_state_dict(self, state_dict):
        for key in self.encoders:
            self.encoders[key].upgrade_state_dict(state_dict)
        return state_dict
