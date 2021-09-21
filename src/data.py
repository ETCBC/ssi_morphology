"""Functions and constants related to Datasets.

MAX_LENGTH          maximum length in tokens of a sequence.
SOS_token           start of sentence token
EOS_token           end of sentence token
INPUT_WORD_TO_IDX   mapping from an input token to an index
OUTPUT_WORD_TO_IDX  mapping from an output token to an index

mc_reduce           reduce an output sequence to a more compact form
mc_expand           inverse of the above

collate_fn          Create a batch from a list of input records

encode_string       Convert a string to a Torch Tensor, using a mapping
decode_string       Convert a Torch Tensor to a string, using a mapping

Dataset wrappers:
    HebrewVerses    a pytorch Dataset around the hebrew bible, returns verses.
    HebrewWords     a pytroch Dataset arount the hebrew bible, returns words.

"""
import collections
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import device

MAX_LENGTH = 25

MC_PREFIXES = ['!', ']', '@']

SOS_token = 0
EOS_token = 1

INPUT_WORD_TO_IDX = {
        'SOS': SOS_token,
        'EOS': EOS_token,
        ' ': 2,
        '*': 3,
        '.': 4,
        ':': 5,
        ';': 6,
        '<': 7,
        '>': 8,
        '@': 9,
        'A': 10,
        'B': 11,
        'C': 12,
        'D': 13,
        'E': 14,
        'F': 15,
        'G': 16,
        'H': 17,
        'I': 18,
        'J': 19,
        'K': 20,
        'L': 21,
        'M': 22,
        'N': 23,
        'O': 24,
        'P': 25,
        'Q': 26,
        'R': 27,
        'S': 28,
        'T': 29,
        'U': 30,
        'V': 31,
        'W': 32,
        'X': 33,
        'Y': 34,
        'Z': 35
    }

OUTPUT_WORD_TO_IDX = {
        'SOS': SOS_token,
        'EOS': EOS_token,
        ' ': 2,
        '!': 3,
        '&': 4,
        '(': 5,
        '+': 6,
        '-': 7,
        '/': 8,
        ':': 9,
        '<': 10,
        '=': 11,
        '>': 12,
        '[': 13,
        ']': 14,
        '_': 15,
        '~': 16,
        'a': 17,
        'B': 18,
        'c': 19,
        'C': 20,
        'd': 21,
        'D': 22,
        'F': 23,
        'G': 24,
        'H': 25,
        'J': 26,
        'K': 27,
        'L': 28,
        'M': 29,
        'n': 30,
        'N': 31,
        'o': 32,
        'p': 33,
        'P': 34,
        'Q': 35,
        'R': 36,
        'S': 37,
        'T': 38,
        'u': 39,
        'V': 40,
        'W': 41,
        'X': 42,
        'Y': 43,
        'Z': 44
    }


class HebrewVerses(Dataset):
    """A Pytorch wrapper around the hebrew bible text. Processed per verse."""

    def __init__(self, input_filename: str, output_filename: str,
                 transform=None):
        """
        Args:
            input_filename (str)
            output_filename (str)
            transform (callable, optional): Optional transform to be applied
                            on a sample.

        The files are formatted as one verse per line, with tab separated
        metadata: book chapter verse text

        Note: output is reduced using the mc_reduce function

        The dataset contains hashes with:
            book: str
            chapter: str
            verse: int
            text: str
            output: str
            encoded_text: Tensor
            encoded_output: Tensor
        """
        with open(input_filename, 'r') as f:
            self.input_data = f.readlines()

        with open(output_filename, 'r') as f:
            self.output_data = f.readlines()

        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # data properties from the intput
        bo, ch, ve, text = tuple(self.input_data[idx].strip().split('\t'))
        sample = {
                "book": bo,
                "chapter": int(ch),
                "verse": int(ve),
                "text": text,
                "encoded_text": encode_string(text, INPUT_WORD_TO_IDX, add_sos=False, add_eos=True)
                }

        # data properties from the output
        bo, ch, ve, text = tuple(self.output_data[idx].strip().split('\t'))

        # reduce output to simpler form
        text = mc_reduce(text)

        sample["output"] = text
        sample["encoded_output"] = encode_string(text, OUTPUT_WORD_TO_IDX, add_sos=True, add_eos=True)

        return sample


class HebrewWords(Dataset):
    """A Pytorch wrapper around the hebrew bible text. Processed per word."""

    def __init__(self, input_filename: str, output_filename: str, 
                 sequence_length: int, transform=None):
        """
        Args:
            input_filename (str)
            output_filename (str)
            sequence_length (int): number of words in single training sample
            transform (callable, optional): Optional transform to be applied
                            on a sample.

        The files are formatted as one verse per line, with tab separated
        metadata: book chapter verse text

        Note: output is reduced using the mc_reduce function

        The dataset contains hashes with:
            text: str
            output: str
            encoded_text: Tensor
            encoded_output: Tensor
        """
        with open(input_filename, 'r') as f:
            input_verses = f.readlines()

        with open(output_filename, 'r') as f:
            output_verses = f.readlines()

        assert(len(input_verses) == len(output_verses))
        


        self.input_data = []
        self.output_data = []

        all_input_words = []
        all_output_words = []

        for i in range(len(input_verses)):
            bo, ch, ve, text = tuple(input_verses[i].strip().split('\t'))
            bo, ch, ve, output = tuple(output_verses[i].strip().split('\t'))

            input_words = text.split()
            output_words = re.split('_| ', output)
            
            if (len(input_words) == len(output_words)):
                all_input_words += input_words
                all_output_words += output_words
            else:
                print(f"Encoding issue with {bo} {ch} {ve} : mismatch in number of words")
                print(input_words)
                print(output_words)

        for heb_word in range(len(all_input_words)+1):
            input_seq = ' '.join([all_input_words[ind % len(all_input_words)] for ind in range(heb_word, heb_word + sequence_length)])
            output_seq = ' '.join([all_output_words[ind % len(all_output_words)] for ind in range(heb_word, heb_word + sequence_length)])
            
            # Add if not cases of ketiv-qere
            if '*' not in input_seq:
                self.input_data.append(input_seq)
                self.output_data.append(output_seq)

        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # data properties from the intput
        text = self.input_data[idx]
        sample = {
                "text": text,
                "encoded_text": encode_string(text, INPUT_WORD_TO_IDX, add_sos=False, add_eos=True)
                }

        # data properties from the output
        text = self.output_data[idx]

        # reduce output to simpler form
        text = mc_reduce(text)

        sample["output"] = text
        sample["encoded_output"] = encode_string(text, OUTPUT_WORD_TO_IDX, add_sos=True, add_eos=True)

        return sample

    def make_longer_sequences(self, data_list, sequence_length):

        data_list = []
        for w in range(len(data_list)+1):
            seq = ' '.join([data_list[ind % len(data_list)] for ind in range(w, w + sequence_length)])
            all_sequences.append(seq)
         

        


def collate_fn(batch):
    """Collate (combine) several records into a single tensor.

    The decoder_input and decoder_target are truncated and aligned
    such that no needless work is done.

    Returns:
        encoder_input: torch.Tensor[Ti, B]
        encoder_lengths: [B]

        decoder_input: torch.Tensor[To, B]
        decoder_target: torch.Tensor[To, B]
        decoder_lengths: [B]
    where:
        B is batch size
        Ti is length of longest input sequence
        To is length of longest output sequence
    """
    # Encoder input
    encoder_input = [b['encoded_text'] for b in batch]
    encoder_input = pad_sequence(encoder_input)
    encoder_lengths = [b['encoded_text'].size()[0] for b in batch]

    # Decoder input
    # the input for the decoder, truncated so that we do not predict
    # anything for the final EOS_token
    decoder_input = [b['encoded_output'][:-1] for b in batch]
    decoder_input = pad_sequence(decoder_input)

    # Decoder target
    # the output for the decoder, shifted such that the first output
    # corresponds to input of the SOS_token
    decoder_target = [b['encoded_output'][1:] for b in batch]
    decoder_target = pad_sequence(decoder_target)
    decoder_lengths = [b['encoded_output'].size()[0] - 1 for b in batch]

    return encoder_input, encoder_lengths, decoder_input, decoder_target, decoder_lengths


def encode_string(seq: str, d: dict, add_sos=False, add_eos=True):
    """Convert a string to a Tensor with indices using the given dictionary.

    If add_sos adds an SOS_token at the start of the sentence (default=False)
    If add_eos (default) adds an EOS_token at the end of the sentence.
    """
    idxs = []
    if add_sos:
        idxs.append(SOS_token)

    for w in seq:
        idxs.append(d[w])

    if add_eos:
        idxs.append(EOS_token)
    return torch.tensor(idxs, dtype=torch.long, device=device)


def decode_string(t, d:dict, strip_sos=True, strip_eos=True):
    """Convert a Tensor with indices to a string using the given dictionary.

    If strip_eos (default), removes all EOS_tokens from the string.
    If strip_sos (default), removes all SOS_tokens from the string.
    """
    inv_d = {v: k for k, v in d.items()}
    seq = ""
    for c in list(t):
        if isinstance(c, torch.Tensor):
            c = c.item()

        if strip_eos and c == EOS_token:
            continue

        if strip_sos and c == SOS_token:
            continue

        seq = seq + inv_d[c]
    return seq


def mc_reduce(s: str) -> str:
    """Reduce the output to a minimal form.

    The reduction consists of removing the left-most marker from all
    the doubly marked prefixes and the redundant colon of the vowel
    pattern mark.
    """
    for c in MC_PREFIXES:
        s = re.sub(f'{c}([^{c}]*{c})', r'\1', s)
    return s.replace(':', '')


def mc_expand(s: str) -> str:
    """
    This function undoes the reduction. The hyphen in the search pattern
    makes sure that we stay within a single analytical word.
    """

    s = re.sub(r'([a-z]+)', r':\1', s)
    r = re.sub('(.)', r'\\\1', ''.join(MC_PREFIXES))
    for c in MC_PREFIXES:
        s = re.sub(f'([^-{r}]*{c})', f'{c}\\1', s)
    return s


def read_data_from_file(filename: str):
    """Read data from a text file and return a Dict.

    The file is one verse per line, tab separated with some metadata:
    book chapter verse text

    The function returns a dictionary indexed by 'book',
    containing a list of all words.
    """

    data_dict = collections.defaultdict(list)

    with open(filename) as fp:
        line = fp.readline()
        while line:
            bo, ch, ve, text = tuple(line.strip().split('\t'))
            words = text.split()
            for w in words:
                # in the output data, composite placenames have a '_',
                # which cannot be found in the input data
                words_split = w.split('_')
                for word_split in words_split:
                    data_dict[bo].append(word_split)

            line = fp.readline()

    return data_dict
