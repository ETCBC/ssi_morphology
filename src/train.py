import sys
from signal import signal, SIGINT, SIG_DFL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import time
from .config import check_abort, abort_handler
from .data import HebrewWords, collate_fn
from .data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX, decode_string
from .model import HebrewEncoder, HebrewDecoder, save_encoder_decoder
from .evaluate import score, score_batch


# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def train(training_data=None, evaluation_data=None,
          encoder=None, decoder=None,
          loss_function=None, log_dir=None,
          max_epoch=100, torch_seed=42, learning_rate=0.1, batch_size=20):
    # make the runs as repeatable as possible
    torch.manual_seed(torch_seed)

    # Tell Python to run the abort_handler() function when SIGINT is recieved
    signal(SIGINT, abort_handler)

    # tensorboard
    writer = SummaryWriter(log_dir)

    timer_start = time.time()
    timer = timer_start
    counter = 0
    max_accuracy = 0.0
    for epoch in range(max_epoch):
        if check_abort():
            break

        for encoder_input, encoder_lengths, decoder_input, decoder_target, decoder_lengths in training_data:
            if check_abort():
                break

            counter += batch_size
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # go over the input sequence
            encoder_output, encoder_hidden = encoder(encoder_input, lengths=encoder_lengths)

            # take over with the decoder and let it run over the output sequence
            decoder_hidden = encoder_hidden

            decoder_output, decoder_hidden = decoder(decoder_input, hidden=decoder_hidden, lengths=decoder_lengths)

            # pack the target tokens in the same way we'll get the output
            decoder_target = pack_padded_sequence(decoder_target, decoder_lengths, enforce_sorted=False)

            # calcualte the loss
            loss += loss_function(
                    decoder_output.data.view(-1, decoder.output_dim),
                    decoder_target.data
                    )

            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            # every N steps, print some diagnostics
            if counter % 10000 == 0:
                oldtimer = timer
                timer = time.time()
                sentence = encoder_input[:, 0].view(-1)  # take the first sentence
                sentence = sentence[0:encoder_lengths[0]]  # trim padding
                sentence = decode_string(sentence, INPUT_WORD_TO_IDX)

                gold = decoder_input[:, 0].view(-1)  # take the first sentence
                gold = gold[1:decoder_lengths[0]]  # trim padding and SOS
                gold = decode_string(gold, OUTPUT_WORD_TO_IDX)

                system, system_lengths = pad_packed_sequence(decoder_output)  # (To, B, H)
                system = torch.argmax(system, dim=2)  # (To, B)
                system = system[:, 0]  # take first sentence
                system = system[0:system_lengths[0]]  # remove padding
                system = decode_string(system, OUTPUT_WORD_TO_IDX)

                # TODO: is NLLLoss averaged or summed?
                print(f'step= {counter} epoch= f{epoch} t={timer - timer_start} dt={timer - oldtimer} batchloss={loss.item()}')

                print(f'\tverse: {sentence}\tgold: {gold}\tsystem: {system}')
                writer.add_text('sample', sentence + "<=>" + system, global_step=counter)
                writer.add_scalar('Loss/train', loss.item(), global_step=counter)

                # for all parameters, write the mean to tensorboard
                for name, val in encoder.named_parameters():
                    writer.add_scalar('encoder.' + name, torch.mean(val), global_step=counter)
                for name, val in decoder.named_parameters():
                    writer.add_scalar('decoder.' + name, torch.mean(val), global_step=counter)

        # per epoch evaluation
        oldtimer = time.time()
        results = score_batch(encoder, decoder, evaluation_data)
        timer = time.time()

        writer.add_scalar('Eval/accuracy', results['accuracy'], global_step=counter)
        max_accuracy = max(max_accuracy, results['accuracy'])
        print('\n\n')
        print('Evaluation===================================================')
        print(f'epoch={epoch} eval={timer - oldtimer} accuracy={results["accuracy"]} best={max_accuracy}')
        print('=============================================================')
        print('\n')
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
        "hparam/accuracy": max_accuracy
    })
    writer.close()

    # Restore default behaviour for Ctrl-c
    signal(SIGINT, SIG_DFL)


if __name__ == '__main__':
    # network and training settings
    batch_size = 20
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

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(evaluation_data, batch_size=50, shuffle=False, collate_fn=collate_fn)
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
    log_dir = 'runs/lll2'

    # optimization strategy
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # print info to stdout
    summary(encoder)
    summary(decoder)

    train(training_data=training_loader,
          evaluation_data=eval_loader,
          encoder=encoder,
          decoder=decoder,
          loss_function=loss_function,
          log_dir=log_dir,
          torch_seed=torch_seed,
          learning_rate=learning_rate,
          batch_size=batch_size
          )
