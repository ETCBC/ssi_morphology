import sys
from signal import signal, SIGINT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchinfo import summary
import time
from data import HebrewBible, MAX_LENGTH, SOS_token
from data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX
from model import HebrewEncoder, HebrewDecoder, device
from evaluate import evaluate
from torch.utils.tensorboard import SummaryWriter

abort = False  # Global variable to catch Ctrl-C


def handler(signal_received, frame):
    global abort
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    abort = True


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def train(training_data, evaluation_data,
          max_epoch=100, torch_seed=42, hidden_dim=64, learning_rate=0.1):
    # make the runs as repeatable as possible
    torch.manual_seed(torch_seed)

    encoder = HebrewEncoder(input_dim=len(INPUT_WORD_TO_IDX), hidden_dim=hidden_dim)
    decoder = HebrewDecoder(hidden_dim=hidden_dim, output_dim=len(OUTPUT_WORD_TO_IDX))

    loss_function = nn.NLLLoss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # tensorboard
    log_dir = f'runs/{hidden_dim}hidden_{torch_seed}seed_{learning_rate}lr'
    writer = SummaryWriter(log_dir)

    # print info to stdout
    summary(encoder)
    summary(decoder)

    timer = time.time()
    counter = 0
    for epoch in range(max_epoch):
        if abort:
            break

        for verse in training_data:
            if abort:
                break

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

            # every 50 steps, print some diagnostics
            if counter % 50 == 0:
                oldtimer = timer
                timer = time.time()
                output = evaluate(encoder, decoder, verse['text'])
                print(counter, epoch, timer - oldtimer,  loss.item() / output_length)
                print('verse:  ', verse['text'])
                print('gold:   ', verse['output'])
                print('system: ', output)
                print(' -- ')
                writer.add_text('sample', verse['text'] + "\n" + output, global_step=counter)
                writer.add_scalar('Loss/train', loss.item(), global_step=counter)

                # for all parameters, write the mean to tensorboard
                for name, val in encoder.named_parameters():
                    writer.add_scalar('encoder.' + name, torch.mean(val), global_step=counter)
                for name, val in decoder.named_parameters():
                    writer.add_scalar('decoder.' + name, torch.mean(val), global_step=counter)

    # write summary to tensorboard
    writer.add_hparams({
        "hidden_dim": hidden_dim,
        "torch_seed": torch_seed,
        "learning_rate": learning_rate,
        "loss_function": type(loss_function).__name__,
        "optimizer_encoder": type(encoder_optimizer).__name__,
        "optimizer_decoder": type(decoder_optimizer).__name__
    }, {
        "hparam/loss": 0.0
    })
    writer.close()


if __name__ == '__main__':
    # load the dataset, and split 70/30 in test/eval
    bible = HebrewBible('data/t-in_voc', 'data/t-out')
    len_train = int(0.7 * len(bible))
    len_eval = len(bible) - len_train
    training_data, evaluation_data = random_split(
            bible, [len_train, len_eval], generator=torch.Generator().manual_seed(42))

    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    train(training_data, evaluation_data)
