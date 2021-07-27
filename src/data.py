"""Functions and constants related to Datasets.

MAX_LENGTH          maximum length in tokens of an input sequence.
SOS_token           start of sentence token
EOS_token           end of sentence token
INPUT_WORD_TO_IDX   mapping from an input token to an index
OUTPUT_WORD_TO_IDX  mapping from an output token to an index

mc_reduce           reduce an output sequence to a more compact form
mc_expand           inverse of the above

encode_string       Convert a string to a Torch Tensor, using a mapping
decode_string       Convert a Torch Tensor to a string, using a mapping

Dataset wrappers:
    HebrewBible     a pytorch Dataset around the hebrew bibble, returns verses.

"""
import collections
import re
import torch
from torch.utils.data import Dataset
from config import device

MAX_LENGTH = 500

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


class HebrewBible(Dataset):
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
                "encoded_text": encode_string(text, INPUT_WORD_TO_IDX)
                }

        # data properties from the output
        bo, ch, ve, text = tuple(self.output_data[idx].strip().split('\t'))

        # reduce output to simpler form
        text = mc_reduce(text)

        sample["output"] = text
        sample["encoded_output"] = encode_string(text, OUTPUT_WORD_TO_IDX)

        return sample


def encode_string(seq: str, d: dict, add_eos=True):
    """Convert a string to a Tensor with indices using the given dictionary.

    If add_eos (default) adds an EOS_token at the end of the sentence.
    """
    idxs = [d[w] for w in seq]
    if add_eos:
        idxs.append(EOS_token)
    return torch.tensor(idxs, dtype=torch.long, device=device)


def decode_string(t, d:dict, strip_eos=True):
    """Convert a Tensor with indices to a string using the given dictionary.

    If strip_eos (default), removes all EOS_tokens from the string.
    """
    inv_d = {v: k for k, v in d.items()}
    seq = ""
    for c in list(t):
        if isinstance(c, torch.Tensor):
            c = c.item()

        if strip_eos:
            if c != EOS_token:
                seq = seq + inv_d[c]
        else:
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


def read_data_from_file(filename: str) -> dict[str, list]:
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
