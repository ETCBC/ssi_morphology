"""Functions to evaluate a trained model.

score     evaluate a trained encoder/decoder on a dataset
"""
import heapq
import torch
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np

from .config import device, check_abort
from .data import decode_string
from .data import OUTPUT_WORD_TO_IDX, SOS_token, EOS_token
from .data import MAX_LENGTH
from .model import reshape_hidden


class Beam():
    """A partially decoded sequence.

    An object holding a partially decoded sequence and the hidden state of
    the decoder.
    A beam can branch(score, token, state), and you can inspect the last
    element of the beam with beam.last()
    """
    def __init__(
            self,
            score=0.0,
            state=None,
            sequence=None,
            length=None,
            max_length=MAX_LENGTH
            ):
        self.max_length = max_length  # maximum length of a sequence
        self.state = state  # the hidden state of the decoder
        self.score = score   # log of sequence probability

        # the decoded sequence
        if sequence is not None and length is not None:
            self.sequence = sequence
            self.length = length
        else:
            self.sequence = np.zeros(self.max_length)
            self.sequence[0] = SOS_token
            self.length = 1  # the length of the decoded sequence

    def branch(self, score=0, token=None, state=None):
        """return a copy of this sequence without the hidden state"""
        assert self.length < self.max_length
        new_beam = Beam(
                state=state,
                score=self.score - score,
                sequence=self.sequence.copy(),
                length=self.length + 1,
                max_length=self.max_length
                )
        new_beam.sequence[self.length] = token
        return new_beam

    def last(self):
        """return the last token in the sequence"""
        return self.sequence[self.length - 1]

    def __lt__(self, other):
        """for ordering beams in the heap"""
        if self.score == other.score:
            return self.length < other.length
        return self.score < other.score

    def __repr__(self):
        pretty = decode_string(self.sequence[0:self.length], OUTPUT_WORD_TO_IDX)
        return f"{self.score} {self.length}" + pretty


def score(encoder, decoder, dataset, max_length=MAX_LENGTH):
    """Evaluate a trained encoder/decoder on a dataset using greedy decoding.

    We evaluate in batches to make the evaluation run faster. This adds some
    complexity, but very similar code is needed later for beamsearch decoding.

    TODO: split running the endocer/decoder and evaluating gold/system to
    different functions, and use better comparision metrics (fi 'badness'
    script.)

    TODO: do not use packed input for the GRU

    Arguments:
        encoder
        decoder
        dataset     A generator for evaluation batches
        max_length  (optional, default=MAX_LENGTH) Safety limit to prevent
                    endless output

        output      A dict containing the following elements:
                    total      number of transcriptions
                    correct    number of correct transcriptions
                    accuracy   percentage of fully correct transcriptions
    """
    total = 0
    correct = 0
    for encoder_input, encoder_lengths, \
            decoder_input, decoder_target, target_lengths in dataset:
        if check_abort():
            break

        # encoder
        # go over the full input sequence with an emtpy hidden state
        _, encoder_hidden = encoder(
                encoder_input, lengths=encoder_lengths)

        # decoder
        # hidden state is initialized from the encoder
        decoder_hidden = reshape_hidden(
                encoder_hidden, encoder.num_layers,
                encoder.D, -1, encoder.hidden_dim)

        # decoder_input
        # as we do greedy decoding, we only need the first SOS_tokens as the
        # second input we continue with the predicted token from the previous
        # step
        batch_size = decoder_input.size()[1]
        decoder_input = decoder_input[0, :].view(1, batch_size)  # [Ti=1, B]
        decoder_lengths = [1] * batch_size
        current_beams = list(range(batch_size))

        # decoder_output
        system = [[] for _ in range(batch_size)]

        for _ in range(max_length):
            # apply a single step with the decoder to all the beams at once
            decoder_output, decoder_hidden = \
                    decoder(decoder_input,
                            hidden=decoder_hidden,
                            lengths=decoder_lengths
                            )
            decoder_output, _ = pad_packed_sequence(decoder_output)
            # [1, B, H], _

            # greedy decoding: find the most likely output
            _, topi = decoder_output.topk(1,  dim=2)  # [1, B, 1]
            topi = topi.view(-1)  # [B]

            # store the tokens and prepare next iteration
            next_beams = []
            decoder_input = []
            next_hidden = []
            for batch_idx, beam_idx in enumerate(current_beams):
                # the predicted next token for this beam
                pred_token = topi[batch_idx].item()

                # store this output at the corresponding beam
                system[beam_idx].append(pred_token)

                # do we need to continue with this beam?
                if pred_token != EOS_token:
                    next_beams.append(beam_idx)
                    decoder_input.append(pred_token)
                    next_hidden.append(
                            decoder_hidden[:, batch_idx, :].unsqueeze(dim=1)
                            )

            # are there any beams to continue?
            if len(next_beams) == 0:
                break

            # prepare input for the next iteration
            decoder_input = torch.tensor(decoder_input).view(1, -1).to(device)
            decoder_hidden = torch.cat((*next_hidden,), dim=1)
            decoder_lengths = [1] * len(next_beams)

            current_beams = next_beams

        # Finished decoding, loop over the beams to compare strings
        for beam_idx, pred_tokens in enumerate(system):
            output = decode_string(
                    pred_tokens, OUTPUT_WORD_TO_IDX,
                    strip_sos=True, strip_eos=True)

            # fully correct transcription
            gold = decode_string(
                decoder_target[0:target_lengths[beam_idx], beam_idx],
                OUTPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True
                )

            total += 1  # total number of transcriptions
            if output == gold:
                correct += 1  # total number of correct transcriptions

    if total > 0:
        accuracy = (1.0 * correct) / (1.0 * total)
    else:
        accuracy = 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }


def print_queue(msg, queue):
    """Print the human readable contents of a queue"""
    for i, beam in enumerate(queue):
        print(
                msg, i,
                'score=', beam.score,
                decode_string(
                    beam.sequence[0:beam.length],
                    OUTPUT_WORD_TO_IDX,
                    strip_sos=False,
                    strip_eos=False
                    )
            )


class BatchedBFBS():
    """A decoding job encapsulating partial and full decoded sequences."""

    def __init__(self, state=None, num_results=5, beam_width=5, max_length=MAX_LENGTH):
        self.is_done = False
        self.is_waiting = False
        self.queue = []
        self.results = []
        self.max_length = max_length
        self.num_results = num_results
        self.beam_width = beam_width
        self.beams_per_length = [0] * (self.max_length + 2)
        if state is not None:
            heapq.heappush(self.queue, Beam(state=state))

    def pre(self):
        """Decode a sequence until we require a step with the network

        Arguments:
            queue   [Beam]    The heapq minqueue containing the beams
            results [Beam]    Possible finished decodings
            beams_per_length [int] of max_length
            num_results int   The number of decodings to find
            beam_width int    The branching factor for a beam
            max_length int    Maximum length of a decoded sequence

        Returns:
            beam    The next Beam to feed through the network
             or
            None    No more beams to process
        """
        while len(self.queue) > 0 and len(self.results) < self.num_results:  # 4, stop condition
            beam = heapq.heappop(self.queue)  # 5

            # only consider beam_width beams of each length
            if self.beams_per_length[beam.length] >= self.beam_width:  # 6
                continue  # 7

            self.beams_per_length[beam.length] += 1  # 8

            if beam.last() == EOS_token:  # 9
                self.results.append(beam)  # 10
                continue

            # beams without an EOS_token that are already maximum length cannot
            # result in a proper sequence, so drop them.
            if beam.length == self.max_length:
                continue

            self.is_waiting = True
            return beam

        self.is_done = True
        return None

    def post(self, beam, decoder_output, decoder_hidden):
        """Postprocess the network output"""

        self.is_waiting = False

        # NOTE: the paper pushes all output to the queue however, the queue
        # then showed up as a CPU bottleneck.  Here, only add the beams that
        # have some chance of contributing.
        #
        # for now, with batch size is 1:
        topv, topi = torch.topk(decoder_output, self.beam_width)

        # bring back to cpu
        topi = topi.cpu().numpy()
        topv = topv.cpu().numpy()

        for token, token_score in zip(topi, topv):  # 12
            # Skip heuristices
            # sh = s + h(x, y + token)  # 14
            heapq.heappush(self.queue, beam.branch(  # 15
                    token_score,
                    token,
                    decoder_hidden)
                    )

        if len(self.queue) == 0 or len(self.results) >= self.num_results:
            self.done = True


def batched_bfbs(
        decoder,
        decoder_hidden,
        num_results=5,
        beam_width=5,
        max_length=MAX_LENGTH
        ):
    job = BatchedBFBS(state=decoder_hidden)

    jobs = []
    batch_size = decoder_hidden.size()[1]  # B
    # decoder hidden is of shape [D=1 * num_layers, batch_size, H_out]
    # decoder output is of shape [L=1, batch_size, D=1 * Hout]

    for batch_idx in range(batch_size):
        jobs.append(BatchedBFBS(
            state=decoder_hidden[:, batch_idx, :].contiguous()
        ))

    # keep running while there are jobs to finish
    while len(jobs) > 0:
        beams_for_network = []

        # get the next input for the network
        for job in jobs:
            if not job.is_done:
                beam = job.pre()
                if beam:
                    beams_for_network.append(beam)

        if len(beams_for_network) == 0:
            break

        # beam.state is of shape [num_layers, 1, H_out]
        # we need to create [num_layers, batch_size, H_out]
        # so drop the batch_size dim,
        # stack along a new dimension
        # and transpose 0 (batch_size) with 1 (num_layers)
        hidden = torch.stack([
            beam.state.squeeze(dim=1) for beam in beams_for_network
            ])
        hidden = torch.transpose(hidden, 0, 1).contiguous()
        # [num_layers, batch_size, H_out]

        # the input must be in shape [L=1, batch_size]
        inpt = [beam.last() for beam in beams_for_network]
        inpt = torch.LongTensor(inpt).to(device).view(1, -1)

        # pass it through the network
        decoder_output, decoder_hidden = decoder(inpt, hidden=hidden)

        # cast the output to python list containing tensors of [H]
        decoder_output = list(decoder_output.view(len(beams_for_network), -1))

        # decoder hidden is of shape [D=1 * num_layers, batch_size, H_out]
        # cast to a python list of [D=1*num_layers, H_out]
        decoder_hidden = list(torch.transpose(decoder_hidden, 0, 1))

        # postprocess output
        for job in jobs:
            if job.is_waiting:
                job.post(
                        beams_for_network.pop(0),
                        decoder_output.pop(0),
                        decoder_hidden.pop(0)
                        )

    results = []
    for job in jobs:
        if len(job.results) > 0:
            best_beam = min(job.results)
            results.append(best_beam.sequence[0:best_beam.length])
        else:
            results.append([SOS_token, EOS_token])

    return results


def best_first_beam_search(
        decoder,
        decoder_hidden,
        beam_width=5,
        num_results=5,
        max_length=MAX_LENGTH
        ):
    """
    Following: Best-First Beam Search (algorithm 2)
    Clara Meister, Tim Vieira, Ryan Cotterell
    https://arxiv.org/abs/2007.03909

    NOTES:
    The score in the paper is monotonously decreasing s(x, y_t) >= s(x, y_t+1),
    and they use a priority queue.
    Our network returns logsoftmax, ie (-inf, 0], which we subtract from a
    starting cost of 0, and we use a minqueue provided by the heapq library.
    """
    queue = []  # 1
    results = []
    beam = Beam(state=decoder_hidden)
    heapq.heappush(queue, beam)  # 2

    # we start a length=1, and we need to count one past max_length
    beams_per_length = [0] * (max_length + 2)

    while len(queue) > 0 and len(results) < num_results:  # 4, stop condition
        beam = heapq.heappop(queue)  # 5

        # only consider beam_width beams of each length
        if beams_per_length[beam.length] >= beam_width:  # 6
            continue  # 7

        beams_per_length[beam.length] += 1  # 8

        if beam.last() == EOS_token:  # 9
            results.append(beam)  # 10
            continue

        # beams without an EOS_token that are already maximum length cannot
        # result in a proper sequence, so drop them.
        if beam.length == max_length:
            continue

        # 11
        decoder_output, decoder_hidden = decoder(  # 13
                torch.LongTensor([[beam.last()]]).to(device),
                hidden=beam.state
                )
        # [1, B, H], _

        # NOTE: the paper pushes all output to the queue however, the queue
        # then showed up as a CPU bottleneck.  Here, only add the beams that
        # have some chance of contributing.
        #
        # for now, with batch size is 1:
        decoder_output = decoder_output.view(-1)  # [H]
        topv, topi = torch.topk(decoder_output, beam_width)

        # bring back to cpu
        topi = topi.cpu().numpy()
        topv = topv.cpu().numpy()

        for token, token_score in zip(topi, topv):  # 12
            # Skip heuristices
            # sh = s + h(x, y + token)  # 14
            heapq.heappush(queue, beam.branch(  # 15
                    token_score,
                    token,
                    decoder_hidden)
                    )

    if len(results) > 0:  # 16
        best_beam = min(results)
        return best_beam.sequence[0:best_beam.length]

    return []


def score_beam_search(encoder, decoder, dataset, max_length=MAX_LENGTH):
    """Evaluate an encoder/decoder on a Dataset using Best First Beam Search.

    Arguments:
        encoder
        decoder
        dataset     A generator for evaluation batches
        max_length  (optional, default=MAX_LENGTH) Safety limit to prevent
                    endless output

        output      A dict containing the following elements:
                    total      number of transcriptions
                    correct    number of correct transcriptions
                    accuracy   percentage of fully correct transcriptions
    """

    total = 0
    correct = 0
    for encoder_input, encoder_lengths, decoder_input, \
            decoder_target, target_lengths in dataset:
        if check_abort():
            break

        # For non-batched beamn search
        # batch_size = decoder_input.size()[1]  # B

        # do a normal encoder pass
        _, encoder_hidden = encoder(encoder_input, lengths=encoder_lengths)

        # decoder_hidden state is initialized from the encoder state
        decoder_hidden = reshape_hidden(  # [num_layers, B, hidden_dim]
                encoder_hidden, encoder.num_layers,
                encoder.D, -1, encoder.hidden_dim)

        system = []
        with torch.no_grad():
            system = batched_bfbs(decoder, decoder_hidden)
            # for non-batched beam search
            # for batch_idx in range(batch_size):
            #     decoder_ouput = batched_bfbs(  # best_first_beam_search(
            #             decoder,
            #             decoder_hidden[:, batch_idx:batch_idx+1, :].contiguous()
            #             )
            #     system.append(decoder_ouput)

        # Finished decoding, loop over the beams to compare strings
        for beam_idx, pred_tokens in enumerate(system):
            # gold input
            # inpt = decode_string(
            #     encoder_input[0:encoder_lengths[beam_idx], beam_idx],
            #     INPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True
            #     )

            # system output
            output = decode_string(
                    pred_tokens,
                    OUTPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True
                    )

            # gold output
            gold = decode_string(
                decoder_target[0:target_lengths[beam_idx], beam_idx],
                OUTPUT_WORD_TO_IDX, strip_sos=True, strip_eos=True
                )

            # print('\n')
            # print('input=', inpt)
            # print('output=', output)
            # print('gold=  ', gold)

            total += 1  # total number of transcriptions
            if output == gold:
                correct += 1  # total number of correct transcriptions
            else:
                pass

    if total > 0:
        accuracy = (1.0 * correct) / (1.0 * total)
    else:
        accuracy = 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy
    }
