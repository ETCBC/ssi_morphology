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
    HebrewWords     a pytroch Dataset arount the hebrew bible, returns words.

"""
import collections
import os
import re

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import device, MC_PREFIXES, PAD_IDX, SOS_token, EOS_token, TRAIN_DATA_FOLDER


class DataReader:
    """
    Read data from input and output file,
    and collect data in groups of sequence_length sequential verses.
    Split data in training, validation and test set.
    """
    def __init__(self, 
                 input_filename: str, 
                 output_filename: str, 
                 sequence_length: int, 
                 val_plus_test_size: float,
                 INPUT_WORD_TO_IDX: dict, 
                 OUTPUT_WORD_TO_IDX: dict):
                    
        self.input_filename = os.path.join(TRAIN_DATA_FOLDER, input_filename)
        self.output_filename = os.path.join(TRAIN_DATA_FOLDER, output_filename)
        self.sequence_length = sequence_length
        self.val_plus_test_size = val_plus_test_size
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX
        
        try:
            with open(self.input_filename, 'r') as f:
                input_verses = f.readlines()
        except FileNotFoundError as err:
            print('Input file missing!')
            print(err)
        try:
            with open(self.output_filename, 'r') as f:
                output_verses = f.readlines()
        except FileNotFoundError as err:
            print('Output file missing!')
            print(err)

        assert len(input_verses) == len(output_verses)

        self.input_data = []
        self.output_data = []
        
        all_input_words_per_book = collections.defaultdict(list)
        all_output_words_per_book = collections.defaultdict(list)
        
        for i in range(len(input_verses)):
            bo, ch, ve, text = tuple(input_verses[i].strip().split('\t'))
            bo, ch, ve, output = tuple(output_verses[i].strip().split('\t'))

            input_words = text.split()
            output_words = output.replace("_", "_ _").split()
            
            if (len(input_words) == len(output_words)):
                all_input_words_per_book[bo].append(input_words)
                all_output_words_per_book[bo].append(output_words)
            else:
                print(f"Encoding issue with {bo} {ch} {ve} : mismatch in number of words")
                print(input_words)
                print(output_words)

        all_input_words_per_book_grouped = self.group_verses(all_input_words_per_book)
        all_output_words_per_book_grouped = self.group_verses(all_output_words_per_book)
        
        all_input_words = self.make_flat_lists(all_input_words_per_book_grouped)
        all_output_words = self.make_flat_lists(all_output_words_per_book_grouped)

        all_input_seq_lists, all_output_seq_lists = self.make_rolling_window_strings(all_input_words, all_output_words)
                    
        self.X_train, X_val_test, self.y_train, y_val_test = train_test_split(all_input_seq_lists, all_output_seq_lists, test_size=self.val_plus_test_size, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=11)
        
        self.X_train = [item for sublist in self.X_train for item in sublist]
        self.y_train = [item for sublist in self.y_train for item in sublist]
        self.X_val = [item for sublist in self.X_val for item in sublist]
        self.y_val = [item for sublist in self.y_val for item in sublist]
        self.X_test = [item for sublist in self.X_test for item in sublist]
        self.y_test = [item for sublist in self.y_test for item in sublist]
        
    def group_verses(self, all_words_per_book: dict):
                     
        grouped_verses_dict = {}
        
        for book, verse_list in all_words_per_book.items():
        
            if len(verse_list) <= self.sequence_length:
                grouped_verses_dict[book] = [verse_list]
            else:
                grouped_verses = [verse_list[idx:idx+self.sequence_length] for idx in range(0, len(verse_list), self.sequence_length)]
                if len(grouped_verses[-1]) < self.sequence_length:
                    last_list = grouped_verses.pop(-1)
                    grouped_verses[-1] += last_list
                grouped_verses_dict[book] = grouped_verses
            
        return grouped_verses_dict
        
    def make_rolling_window_strings(self, all_input_words, all_output_words):
    
        all_input_seq_lists = []
        all_output_seq_lists = []
        
        for bo in all_input_words.keys():
            for word_list_input, word_list_output in zip(all_input_words[bo], all_output_words[bo]):
                input_seq_list = []
                output_seq_list = []
                for word_idx in range(len(word_list_input) - self.sequence_length + 1):
                    input_seq = ' '.join(word_list_input[word_idx:word_idx + self.sequence_length])
                    output_seq = ' '.join(word_list_output[word_idx:word_idx + self.sequence_length])
                    
                    if "*" not in input_seq and "_" not in output_seq:
                        input_seq_list.append(input_seq)
                        output_seq_list.append(output_seq)
                        
                        for char in input_seq:
                            if char not in self.INPUT_WORD_TO_IDX:
                                self.INPUT_WORD_TO_IDX[char] = len(self.INPUT_WORD_TO_IDX)
                        for char in output_seq:
                            if char not in self.OUTPUT_WORD_TO_IDX:
                                self.OUTPUT_WORD_TO_IDX[char] = len(self.OUTPUT_WORD_TO_IDX)
                if len(input_seq_list) > 0:
                    all_input_seq_lists.append(input_seq_list)
                    all_output_seq_lists.append(output_seq_list)
                    
        return all_input_seq_lists, all_output_seq_lists
        
    @staticmethod
    def make_flat_lists(data_dict):
        new_data_dict = {}
        for bo, w_list in data_dict.items():
            flat_w_list = [word for sublist in w_list for word in sublist]
            new_data_dict[bo] = flat_w_list
            
        return new_data_dict


class HebrewWords(Dataset):
    """A Pytorch wrapper around the hebrew bible text. Processed per word."""

    def __init__(self, 
                 input_data: list, 
                 output_data: list,
                 INPUT_WORD_TO_IDX: dict, 
                 OUTPUT_WORD_TO_IDX: dict):
        """
        Args:
            input_data: list contains text sequences (str)
            output_data: list  contains text sequences (str)
            INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX: dict used for conversion from chars to intergers and back.
        The files are formatted as one verse per line, with tab separated
        metadata: book chapter verse text
        Note: output is reduced using the mc_reduce function
        The dataset contains hashes with:
            text: str
            output: str
            encoded_text: Tensor
            encoded_output: Tensor
        """

        self.input_data = input_data
        self.output_data = output_data
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text_input = self.input_data[idx]

        text_output = mc_reduce(self.output_data[idx])

        sample = {
                "text": text_input,
                "encoded_text": encode_string(text_input, self.INPUT_WORD_TO_IDX, add_sos=False, add_eos=True),
                "output": text_output,
                "encoded_output": encode_string(text_output, self.OUTPUT_WORD_TO_IDX, add_sos=True, add_eos=True)
                }

        return sample


# function to collate data samples into batch tesors
def collate_transformer_fn(batch):
  
    src_batch, tgt_batch = [], []
    
    for sample in batch:
        src_batch.append(sample['encoded_text'])
        tgt_batch.append(sample['encoded_output'])

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


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
        
        
class DataMerger:
    """
    Class is used when model is trained on two datasets (Hebrew and Syriac)
    at the same time.
    """
    def __init__(self, 
                 input1: list, 
                 input2: list,
                 output1: list, 
                 output2: list):
        self.input1 = input1
        self.input2 = input2
        self.output1 = output1
        self.output2 = output2

    def merge_data(self):
        self.input_data = self.input1 + self.input2
        self.output_data = self.output1 + self.output2
        return self.input_data, self.output_data

    def shuffle_data(self, input_data, output_data):
        input_data_shuffled, output_data_shuffled = shuffle(input_data, output_data, random_state=0)
        return input_data_shuffled, output_data_shuffled
