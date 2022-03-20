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

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
                
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden
        
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        
        


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
