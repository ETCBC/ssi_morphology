"""Functions to evaluate a trained model.

score     evaluate a trained encoder/decoder on a dataset
"""
import torch
from torch.nn.utils.rnn import pad_packed_sequence

from config import device, check_abort
from data import decode_string
from data import OUTPUT_WORD_TO_IDX, EOS_token
from data import MAX_LENGTH


def score(encoder, decoder, dataset, max_length=MAX_LENGTH):
    """Evaluate a trained encoder/decoder on a dataset using greedy decoding.

    We evaluate in batches to make the evaluation run faster. This adds some
    complexity, but very similar code is needed later for beamsearch decoding.

    TODO: split running the endocer/decoder and evaluating gold/system to
    different functions, and use better comparision metrics (fi 'badness' script.)

    Arguments:
        encoder
        decoder
        dataset     A generator for evaluation batches
        max_length  (optional, default=MAX_LENGTH) Safety limit to prevent endless output

        output      A dict containing the following elements:
                    total      number of transcriptions
                    correct    number of correct transcriptions
                    accuracy   percentage of fully correct transcriptions
    """
    total = 0
    correct = 0
    for encoder_input, encoder_lengths, decoder_input, decoder_target, target_lengths in dataset:
        if check_abort():
            break

        # encoder
        # go over the full input sequence with an emtpy hidden state
        encoder_output, encoder_hidden = encoder(encoder_input, lengths=encoder_lengths)

        # decoder
        # hidden state is initialized from the encoder
        decoder_hidden = encoder_hidden

        # decoder_input
        # as we do greedy decoding, we only need the first SOS_tokens
        # as the second input we continue with the predicted token from the previous step
        batch_size = decoder_input.size()[1]
        decoder_input = decoder_input[0, :].view(1, batch_size)  # [Ti=1, B]
        decoder_lengths = [1] * batch_size
        current_beams = list(range(batch_size))

        # decoder_output
        system = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            # apply a single step with the decoder to all the beams at once
            decoder_output, decoder_hidden = decoder(decoder_input, hidden=decoder_hidden, lengths=decoder_lengths)
            decoder_output, _ = pad_packed_sequence(decoder_output)  # [1, B, H], _

            # greedy decoding: find the most likely output
            topv, topi = decoder_output.topk(1,  dim=2)  # [1, B, 1]
            topi = topi.view(-1)  # [B]

            # store the tokens and prepare next iteration
            next_beams = []
            decoder_input = []
            next_hidden = []
            for cb in range(len(current_beams)):
                # the index of this beam in original batch
                s = current_beams[cb]

                # the predicted next token for this beam
                t = topi[cb].item()

                # store this output at the corresponding beam
                system[s].append(t)

                # do we need to continue with this beam?
                if t != EOS_token:
                    next_beams.append(s)
                    decoder_input.append(t)
                    next_hidden.append(decoder_hidden[:, cb, :].unsqueeze(dim=1))

            # are there any beams to continue?
            if len(next_beams) == 0:
                break

            # prepare input for the next iteration
            decoder_input = torch.tensor(decoder_input).view(1, -1).to(device)
            decoder_hidden = torch.cat((*next_hidden,), dim=1)
            decoder_lengths = [1] * len(next_beams)

            current_beams = next_beams

        # Finished decoding, loop over the beams to compare strings
        for si in range(len(system)):
            output = decode_string(system[si], OUTPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True)

            # fully correct transcription
            gold = decode_string(decoder_target[0:target_lengths[si], si], OUTPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True)

            total += 1  # total number of transcriptions
            if output == gold:
                correct += 1

    if total > 0:
        accuracy = (1.0 * correct) / (1.0 * total)
    else:
        accuracy = 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }
