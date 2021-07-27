import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import time
from data import HebrewBible, MAX_LENGTH, SOS_token
from data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX
from model import HebrewEncoder, HebrewDecoder, device


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def train(training_data, evaluation_data,
          max_epoch=100, torch_seed=42, hidden_dim=64, learning_rate=0.1):
    # make the runs as repeatable as possible
    torch.manual_seed(torch_seed)

    encoder = HebrewEncoder(input_dim=len(INPUT_WORD_TO_IDX), hidden_dim=hidden_dim)
    decoder = HebrewDecoder(hidden_dim=hidden_dim, output_dim=len(OUTPUT_WORD_TO_IDX))

    print(encoder)
    print(decoder)

    loss_function = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    timer = time.time()
    counter = 0
    for epoch in range(max_epoch):
        for verse in training_data:
            counter += 1
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # get the data for this step
            input_seq = verse["encoded_text"]
            output_seq = verse["encoded_output"]

            input_length = input_seq.size(0)
            output_length = output_seq.size(0)

            # go over the input sequence
            encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_dim, device=device)
            encoder_hidden = encoder.initHidden()

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_seq[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # take over with the decoder and let it run over the output sequence
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[SOS_token]], device=device)

            for di in range(output_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += loss_function(decoder_output, output_seq[di].view(-1))
                decoder_input = output_seq[di]  # Use oracle predictions

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            if counter % 50 == 0:
                oldtimer = timer
                timer = time.time()
                print(counter, epoch, timer - oldtimer,  loss.item() / output_length)


if __name__ == '__main__':
    # load the dataset, and split 70/30 in test/eval
    bible = HebrewBible('data/t-in_voc', 'data/t-out')
    len_train = int(0.7 * len(bible))
    len_eval = len(bible) - len_train
    training_data, evaluation_data = random_split(
            bible, [len_train, len_eval], generator=torch.Generator().manual_seed(42))

    train(training_data, evaluation_data)
