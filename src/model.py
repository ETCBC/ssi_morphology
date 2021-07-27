import torch
import torch.nn as nn
import torch.nn.functional as F

from config import device


class HebrewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(HebrewEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.input_embeddings = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2)
        self.to(device)

    def forward(self, input, hidden):
        embedded = self.input_embeddings(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_dim, device=device)


class HebrewDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(HebrewDecoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.output_embeddings = nn.Embedding(output_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.to(device)

    def forward(self, input, hidden):
        output = self.output_embeddings(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, 1, self.hidden_dim, device=device)
