"""Functions to evaluate a trained model.

evaluate  run a trained encoder and decoder on a sentence using greedy decoding
score     evaluate a trained encoder/decoder on a dataset
"""
import torch
from torch.nn.utils.rnn import pad_packed_sequence

from .config import device, check_abort
from .data import encode_string, decode_string
from .data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX, EOS_token
from .data import MAX_LENGTH


def evaluate(encoder, decoder, sentence):
    """Evaluate a single sentence using greedy decoding."""
    with torch.no_grad():
        encoder_input = encode_string(sentence, INPUT_WORD_TO_IDX).view(-1, 1)
        encoder_lengths = [encoder_input.size()[0]]

        # go over the input sequence
        encoder_output, encoder_hidden = encoder(encoder_input, lengths=encoder_lengths)

        # take over with the decoder
        decoder_hidden = encoder_hidden
        decoder_input = encode_string("", OUTPUT_WORD_TO_IDX, add_sos=True, add_eos=False).view(-1, 1)
        decoder_lengths = [decoder_input.size()[0]]

        decoded_tokens = []

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden = decoder(decoder_input, hidden=decoder_hidden, lengths=decoder_lengths)
            decoder_output = decoder_output.data.view(-1)
            topv, topi = decoder_output.topk(1)
            decoded_tokens.append(topi.item())
            if topi.item() == EOS_token:
                break

            decoder_input = topi.squeeze().detach().view(-1, 1)

        return decode_string(decoded_tokens, OUTPUT_WORD_TO_IDX)


def score_batch(encoder, decoder, dataset, max_length=MAX_LENGTH):
    """Evaluate a trained encoder/decoder on a dataset.

    Arguments:
        encoder
        decoder
        dataset     An iterable, with the elements a dict with 'text', and 'output'
        max_length

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

        # go over the input sequence
        encoder_output, encoder_hidden = encoder(encoder_input, lengths=encoder_lengths)

        # take over with the decoder and let it run over the output sequence
        decoder_hidden = encoder_hidden

        # decoder_input
        # as we do greedy decoding, we only need the SOS_tokens
        batch_size = decoder_input.size()[1]
        decoder_input = decoder_input[0, :].view(1, batch_size)  # [Ti=1, B]
        decoder_lengths = [1] * batch_size
        system = [[] for _ in range(batch_size)]
        current_beams = list(range(batch_size))

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, hidden=decoder_hidden, lengths=decoder_lengths)
            decoder_output, _ = pad_packed_sequence(decoder_output)  # (1, B, H), _
            topv, topi = decoder_output.topk(1,  dim=2)  # [1, B, 1]
            topi = topi.view(-1)  # [B]

            # store the tokens and prepare next iteration
            next_beams = []
            decoder_input = []
            next_hidden = []
            for cb in range(len(current_beams)):
                s = current_beams[cb]  # the index of this beam in original batch
                t = topi[cb].item()  # the next token for this beam

                # store this output
                system[s].append(t)

                # do we need to continue with this beam?
                if t != EOS_token:
                    next_beams.append(s)
                    decoder_input.append(t)
                    next_hidden.append(decoder_hidden[:, cb, :].unsqueeze(dim=1))

            # are there any beams to continue?
            if len(next_beams) == 0:
                break

            decoder_input = torch.tensor(decoder_input).view(1, -1).to(device)
            decoder_hidden = torch.cat((*next_hidden,), dim=1)
            decoder_lengths = [1] * len(next_beams)

            current_beams = next_beams

        for si in range(len(system)):
            output = decode_string(system[si], OUTPUT_WORD_TO_IDX, strip_eos=True)

            # fully correct transcription
            shouldbe = decode_string(decoder_target[0:target_lengths[si], si], OUTPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True)

            total += 1  # total number of transcriptions
            if output == shouldbe:
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


def score(encoder, decoder, dataset, max_length=MAX_LENGTH):
    """Evaluate a trained encoder/decoder on a dataset.

    Arguments:
        dataset     An iterable, with the elements a dict with 'text', and 'output'

        output      A dict containing the following elements:
                    total      number of transcriptions
                    correct    number of correct transcriptions
                    accuracy   percentage of fully correct transcriptions
    """
    total = 0
    correct = 0
    for record in dataset:
        if check_abort():
            break
        sentence = record['text']
        target = record['output']
        output = evaluate(encoder, decoder, sentence)
        total += 1  # total number of transcriptions
        if output == target:  # fully correct transcription
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
