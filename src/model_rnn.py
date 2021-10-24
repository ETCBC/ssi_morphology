import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from config import device

def reshape_hidden(hidden, num_layers, num_directions, batch, hidden_dim):
    """Reshape the hidden state of an encoder for use in a decoder.
    Arguments
    hidden   The hidden state of the encoder [num_layers * dirs, B, hidden_dim]
    Returns
    hidden   The hidden state of the decoder [num_layers, B, hidden_dim * dirs]
    """
    if num_directions == 1:
        return hidden

    dir_a = hidden.view(num_layers, 2, batch, hidden_dim)[:, 0, :, :]  # [num_layers, batch, hidden_dim]
    dir_b = hidden.view(num_layers, 2, batch, hidden_dim)[:, 1, :, :]  # [num_layers, batch, hidden_dim]

    return torch.cat((dir_a, dir_b), dim=2)

def squash_packed(x, fn, dim=None):
    """Run a function on a PackedSequence.

    This is a shortcut to first unpack the sequence, run the function,
    and repack.

    Arguments

    x           data to call the function on
    fn          Function to call
    dim (None)  If given, assume the function operates on a vector,
                and do fn(x.view(-1, dim))
                Otherwise, do fn(x).

    """
    if dim:
        return PackedSequence(fn(x.data.view(-1, dim)), x.batch_sizes, x.sorted_indices, x.unsorted_indices)
    else:
        return PackedSequence(fn(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices)


class HebrewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, bidir=False):
        """Initialize the encoder.

        input_dim    Size of the input
        hidden_dim   Size of the hidden state
        num_layers   (default=2) Number of layers in the GRU cell
        bidir        (default=False) Apply bi-directional GRU
        """
        super(HebrewEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.bidir = bidir
        self.D = 2 if bidir else 1
        self.input_embeddings = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, self.num_layers, bidirectional=self.bidir)
        self.to(device)

    def forward(self, input, hidden=None, lengths=None):
        """
        input: tensor[Ti, B]
        lengths: tensor[B]
        hidden: tensor[num_layers * dirs, batch_size, hidden_dim]
        """
        embedded = self.input_embeddings(input)
        # embedded: tensor[Ti, B, hidden_dim]
        output = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        # output: PackedSequence
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(self.num_layers * self.D, batch_size, self.hidden_dim, device=device)


class HebrewDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        """Initialize the denoder.

        input_dim    Size of the input
        hidden_dim   Size of the hidden state
        num_layers   (default=2) Number of layers in the GRU cell
        """
        super(HebrewDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.output_embeddings = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.to(device)

    def forward(self, input, hidden=None, lengths=None):
        """
        input: tensor[To, B]
        target: tensor[To, B]
        lengths: tensor[B]
        hidden: tensor[num_layers, batch_size, hidden_dim]
        """
        embedded = self.output_embeddings(input)
        # embedded: tensor[Ti, B, hidden_dim]

        output = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        # output: PackedSequence

        output = squash_packed(output, F.relu)
        output, hidden = self.gru(output, hidden)
        output = squash_packed(output, self.out)
        output = squash_packed(output, self.softmax, dim=self.output_dim)

        return output, hidden

    def initHidden(self, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


def save_encoder_decoder(encoder, decoder, path=None, filename="model.pt"):
    if not path:
        path = "."

    state = {
            "input_dim": encoder.input_dim,
            "hidden_dim": encoder.hidden_dim,
            "output_dim": decoder.output_dim,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict()
            }

    torch.save(state, os.path.join(path, filename))


def load_encoder_decoder(encoder, decoder, path=None, filename="model.pt"):
    if not path:
        path = "."

    state = torch.load(os.path.join(path, filename))

    encoder = HebrewEncoder(state['input_dim'], state['hidden_dim'])
    decoder = HebrewDecoder(state['hidden_dim'], state['output_dim'])

    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])

    return encoder, decoder
