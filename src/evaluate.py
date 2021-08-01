"""Functions to evaluate a trained model.

evaluate  run a trained encoder and decoder on a sentence using greedy decoding
score     evaluate a trained encoder/decoder on a dataset
"""
import torch
from .config import device, check_abort
from .data import MAX_LENGTH, encode_string, decode_string
from .data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX, SOS_token, EOS_token


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    """Evaluate a single sentence using greedy decoding."""
    with torch.no_grad():
        input_tensor = encode_string(sentence, INPUT_WORD_TO_IDX)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(topi.item())
            if topi.item() == EOS_token:
                break

            decoder_input = topi.squeeze().detach()

        return decode_string(decoded_words, OUTPUT_WORD_TO_IDX)


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
        output = evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH)
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
