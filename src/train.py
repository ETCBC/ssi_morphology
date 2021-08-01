import sys
from signal import signal, SIGINT, SIG_DFL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import time
from .config import device, check_abort, abort_handler
from .data import HebrewWords, MAX_LENGTH, SOS_token
from .data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX
from .model import HebrewEncoder, HebrewDecoder, save_encoder_decoder
from .evaluate import evaluate, score


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def train(training_data=None, evaluation_data=None,
          encoder=None, decoder=None,
          loss_function=None, log_dir=None,
          max_epoch=100, torch_seed=42, learning_rate=0.1):
    # make the runs as repeatable as possible
    torch.manual_seed(torch_seed)

    # Tell Python to run the abort_handler() function when SIGINT is recieved
    signal(SIGINT, abort_handler)

    # tensorboard
    writer = SummaryWriter(log_dir)

    timer = time.time()
    counter = 0
    for epoch in range(max_epoch):
        if check_abort():
            break

        for verse in training_data:
            if check_abort():
                break

            counter += 1
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # get the data for this step
            input_seq = verse["encoded_text"]
            output_seq = verse["encoded_output"]

            # go over the input sequence
            encoder_hidden = encoder.initHidden()

            encoder_output, encoder_hidden = encoder(input_seq, encoder_hidden)

            # take over with the decoder and let it run over the output sequence
            decoder_hidden = encoder_hidden

            decoder_output, decoder_hidden = decoder(output_seq[:-1], decoder_hidden)  # ignore output for last token
            loss += loss_function(
                    decoder_output.view(-1, decoder.output_dim),
                    output_seq[1:].view(-1)  # ignore SOS token
                    )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # every N steps, print some diagnostics
            if counter % 500 == 0:
                output_length = output_seq.size(0)
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

        # per epoch evaluation
        results = score(encoder, decoder, evaluation_data)
        writer.add_scalar('Eval/accuracy', results['accuracy'], global_step=counter)
        save_encoder_decoder(encoder, decoder, log_dir, filename=f'model.{epoch}.pt')

    # write summary to tensorboard
    writer.add_hparams({
        "hidden_dim": encoder.hidden_dim,
        "torch_seed": torch_seed,
        "learning_rate": learning_rate,
        "loss_function": type(loss_function).__name__,
        "optimizer_encoder": type(encoder_optimizer).__name__,
        "optimizer_decoder": type(decoder_optimizer).__name__
    }, {
        "hparam/loss": 0.0
    })
    writer.close()

    # Restore default behaviour for Ctrl-c
    signal(SIGINT, SIG_DFL)


if __name__ == '__main__':
    # network and training settings
    hidden_dim = 128
    torch_seed = 42
    learning_rate = 1e-3

    # load the dataset, and split 70/30 in test/eval
    bible = HebrewWords('data/t-in_voc', 'data/t-out')
    len_train = int(0.7 * len(bible))
    len_eval = len(bible) - len_train
    # alwyas use the same seed for train/test split
    training_data, evaluation_data = random_split(
            bible, [len_train, len_eval], generator=torch.Generator().manual_seed(42))

    training_loader = DataLoader(training_data, batch_size=None, shuffle=True)
    print('Training / Eval split: 70/30 using manual_seed=42.')
    print(f'Training size:   {len(training_data)}')
    print(f'Evaluation size: {len(evaluation_data)}')

    # create the network
    encoder = HebrewEncoder(input_dim=len(INPUT_WORD_TO_IDX), hidden_dim=hidden_dim)
    decoder = HebrewDecoder(hidden_dim=hidden_dim, output_dim=len(OUTPUT_WORD_TO_IDX))

    # function to optimize
    loss_function = nn.NLLLoss()

    # log settings
    log_dir = f'runs/{hidden_dim}hidden_{torch_seed}seed_{learning_rate}lr'

    # optimization strategy
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # print info to stdout
    summary(encoder)
    summary(decoder)

    train(training_data=training_loader,
          evaluation_data=evaluation_data,
          encoder=encoder,
          decoder=decoder,
          loss_function=loss_function,
          log_dir=log_dir,
          torch_seed=torch_seed,
          learning_rate=learning_rate
          )
