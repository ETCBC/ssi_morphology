import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import device


class HebrewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HebrewEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.input_embeddings = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2)
        self.to(device)

    def forward(self, input, hidden):
        embedded = self.input_embeddings(input).view(-1, 1, self.hidden_dim)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_dim, device=device)


class HebrewDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(HebrewDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.output_embeddings = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=2)
        self.to(device)

    def forward(self, input, hidden):
        output = self.output_embeddings(input).view(-1, 1, self.hidden_dim)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_dim, device=device)


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
