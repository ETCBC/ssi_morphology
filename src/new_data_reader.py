"""Read and prepare new data on which a model can make predictions."""

import json
import os

from torch.utils.data import Dataset
import torch

from config import PREDICTION_DATA_FOLDER, MODEL_PATH
from data import encode_string
from model_transformer import Seq2SeqTransformer
import yaml

class NewDataReader:
    """
    Read new data from file and make sequences.
    """
    def __init__(self, input_filename, seq_len):

        self.input_filename = input_filename
        self.seq_len = seq_len
        self.input_verses = self.import_data()
        self.data_ids2text = {}
        self.data_ids2labels = {}
        self.prepared_data = {}
        
        self.read_data()
        self.make_sequences()

    def import_data(self):
        try:
            with open(os.path.join(PREDICTION_DATA_FOLDER, self.input_filename), 'r') as f:
                return f.readlines()
        except FileNotFoundError:
            print('Input file missing!')
            
    def read_data(self):
        word_id = 0
        for verse in self.input_verses:
            bo, ch, ve, text = tuple(verse.strip().split('\t'))
            split_text = text.split()

            for word in split_text:
                self.data_ids2text[word_id] = word
                self.data_ids2labels[word_id] = [bo, ch, ve]
                word_id += 1

    def make_sequences(self):
        """
        Make partly overlapping sequences of length seq_len.
        """
        for idx in self.data_ids2text.keys():
            if idx + self.seq_len -1 in self.data_ids2text:
                self.prepared_data[tuple(range(idx, idx+self.seq_len))] = ' '.join([self.data_ids2text[idx] for idx in range(idx, idx+self.seq_len)])


class HebrewWordsNewText(Dataset):
    """A Pytorch wrapper around text. Processed per sequence."""

    def __init__(self, 
                 data: dict,
                 data_labels: dict,
                 INPUT_WORD_TO_IDX: dict, 
                 OUTPUT_WORD_TO_IDX: dict):
        """
        Args:
            input_filename (str)
        Note: output is reduced using the mc_reduce function
        The dataset contains hashes with:
            text: str
            output: str
            encoded_text: Tensor
            encoded_output: Tensor
        """

        self.word_indices = list(data.keys())
        self.word_texts = list(data.values())
        self.data_labels = data_labels
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX
        self.OUTPUT_IDX_TO_WORD = {v:k for k,v in self.OUTPUT_WORD_TO_IDX.items()}

    def __len__(self):
        return len(self.word_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.word_texts[idx]
        sample = {
                'text': text,
                'encoded_text': encode_string(text, self.INPUT_WORD_TO_IDX, add_sos=False, add_eos=True),
                'indices': self.word_indices[idx],
                'labels': [self.data_labels[word_idx] for word_idx in self.word_indices[idx]]
                }
        return sample
        
        
class ConfigParser:
    """
    Parser for configuration file for predictions
    on new data.
    """
    def __init__(self, yaml_file_name):
        self.yaml_file_name = yaml_file_name 
        self.parsed_yaml = self.parse_yaml()
        self.model_folder = None
        self.model_config_file_name = None
        self.model_name = None
        self.new_data_file = None
        self.output = None
        self.predict_idx = None
        self.beam_size = None
        self.beam_alpha = None
        
        self.get_file_names()
        self.model_config_data = self.import_model_config()
        
    def parse_yaml(self):
        try:
            with open(os.path.join(PREDICTION_DATA_FOLDER, self.yaml_file_name), 'r') as f:
                parsed_yaml=yaml.safe_load(f)
            return parsed_yaml
        except FileNotFoundError as err:
            print('No Yaml file found')
            print(err)
    
    def get_file_names(self):
        try:
            model_info = self.parsed_yaml['model_info']
            self.model_folder = model_info['folder']
            self.model_config_file_name = model_info['model_config']
            self.model_name = model_info['model']
            self.output = self.parsed_yaml.get('output')
            self.predict_idx = int(self.parsed_yaml.get('predict_idx', 0))
            self.beam_size = self.parsed_yaml.get('beam_size', 3)
            self.beam_alpha = float(self.parsed_yaml.get('beam_alpha', 0.75))
        except KeyError as err:
            print()
            print(err)
        try:
            self.new_data_file = self.parsed_yaml['new_data']
        except KeyError as err:
            print('No new data file in yaml')
            print(err)
        
    def import_model_config(self):
        try:
            with open(os.path.join(MODEL_PATH, self.model_folder, self.model_config_file_name)) as config_json:
                model_config_data = json.load(config_json)
            return model_config_data
        except FileNotFoundError as err:
            print('Model config file not found!')
            print(err)
  
class ModelImporter:
    """"""
    def __init__(self, config, folder_name, model_name):
        self.config = config
        self.folder_name = folder_name
        self.model_name = model_name
        
        self.num_encoder_layers = None
        self.num_decoder_layers = None
        self.emb_size = None
        self.nhead = None
        self.src_vocab_size = None
        self.tgt_vocab_size = None
        self.ffn_hid_dim = None
        
        self.get_model_hyperparameters()
        self.loaded_transformer = self.load_model(self.config)
        
    def get_model_hyperparameters(self):
        try:
            self.num_encoder_layers = self.config['num_encoder_layers']
            self.num_decoder_layers = self.config['num_decoder_layers']
            self.emb_size = self.config['emb_size']
            self.nhead = self.config['nhead']
            self.src_vocab_size = self.config['src_vocab_size']
            self.tgt_vocab_size = self.config['tgt_vocab_size']
            self.ffn_hid_dim = self.config['ffn_hid_dim']
        except KeyError as err:
            print('Cannot find the model hyperparaqmeters in the config')
            print(err)
        
    def load_model(self, config):
        try:
            loaded_transformer = Seq2SeqTransformer(self.num_encoder_layers, 
                                               self.num_decoder_layers, 
                                               self.emb_size,
                                               self.nhead, 
                                               self.src_vocab_size, 
                                               self.tgt_vocab_size, 
                                               self.ffn_hid_dim
                                               )
            loaded_transformer.load_state_dict(torch.load(os.path.join(MODEL_PATH, self.folder_name, self.model_name)))
            return loaded_transformer
        except FileNotFoundError:
            print('Model file not found!')
            print(os.path.join(self.folder_name, self.model_name))
        