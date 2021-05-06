import torch.nn as nn
import torch
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models import register_model_architecture


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model('rnn_classifier')
class FairseqRNNClassifier(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add a new command-line argument to configure the
        # dimensionality of the hidden state.
        parser.add_argument(
            '--hidden-dim', type=int, metavar='N',
            help='dimensionality of the hidden state',
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a FairseqRNNClassifier instance.

        # Initialize our RNN module
        rnn = RNN(
            # We'll define the Task in the next section, but for now just
            # notice that the task holds the dictionaries for the "source"
            # (i.e., the input sentence) and "target" (i.e., the label).
            input_size=len(task.source_dictionary),
            hidden_size=args.hidden_dim,
            output_size=len(task.target_dictionary),
        )

        # Return the wrapped version of the module
        return FairseqRNNClassifier(
            rnn=rnn,
            input_vocab=task.source_dictionary,
        )

    def __init__(self, rnn, input_vocab):
        super(FairseqRNNClassifier, self).__init__()

        self.rnn = rnn
        self.input_vocab = input_vocab

        # The RNN module in the tutorial expects one-hot inputs, so we can
        # precompute the identity matrix to help convert from indices to
        # one-hot vectors. We register it as a buffer so that it is moved to
        # the GPU when ``cuda()`` is called.
        self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))

    def forward(self, src_tokens, src_lengths):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch. We'll define the Task in the next section, but for
        # now just know that *src_tokens* has shape `(batch, src_len)` and
        # *src_lengths* has shape `(batch)`.
        bsz, max_src_len = src_tokens.size()

        # Initialize the RNN hidden state. Compared to the original PyTorch
        # tutorial we'll also handle batched inputs and work on the GPU.
        hidden = self.rnn.initHidden()
        hidden = hidden.repeat(bsz, 1)  # expand for batched inputs
        hidden = hidden.to(src_tokens.device)  # move to GPU

        for i in range(max_src_len):
            # WARNING: The inputs have padding, so we should mask those
            # elements here so that padding doesn't affect the results.
            # This is left as an exercise for the reader. The padding symbol
            # is given by ``self.input_vocab.pad()`` and the unpadded length
            # of each input is given by *src_lengths*.

            # One-hot encode a batch of input characters.
            input = self.one_hot_inputs[src_tokens[:, i].long()]

            # Feed the input to our RNN.
            output, hidden = self.rnn(input, hidden)

        # Return the final output state for making a prediction
        return output


# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'rnn_classifier'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('rnn_classifier', 'pytorch_tutorial_rnn')
def pytorch_tutorial_rnn(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.hidden_dim = getattr(args, 'hidden_dim', 128)
