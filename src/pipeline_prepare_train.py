import json
import os

import torch
from torch.utils.data import DataLoader

from data import collate_transformer_fn, DataMerger, DataReader, HebrewWords
from transformer_train_fns import initialize_transformer_model, train_transformer
from evaluate_transformer import evaluate_transformer_model

        
class PipeLinePrepare:
    def __init__(self, 
                 input_file: str, 
                 output_file: str,
                 input_file2: str, 
                 output_file2: str, 
                 length: int, 
                 INPUT_WORD_TO_IDX: dict, 
                 OUTPUT_WORD_TO_IDX: dict,
                 batch_size: int, 
                 val_plus_test_size: int
                 ):
        self.input_file = input_file
        self.output_file = output_file
        self.input_file2 = input_file2 
        self.output_file2 = output_file2
        self.length = length
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX
        self.batch_size = batch_size
        self.val_plus_test_size = val_plus_test_size
        self.model_path_full = None

        self.data_set_one = DataReader(input_file, output_file, length, self.val_plus_test_size, INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX)
        
        if self.input_file2 and self.output_file2:
            self.data_set_two = DataReader(self.input_file2, self.output_file2, length, self.val_plus_test_size, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
            self.INPUT_WORD_TO_IDX = self.data_set_two.INPUT_WORD_TO_IDX
            self.OUTPUT_WORD_TO_IDX = self.data_set_two.OUTPUT_WORD_TO_IDX
            
        self.INPUT_IDX_TO_WORD = {v:k for k,v in self.INPUT_WORD_TO_IDX.items()}
        self.OUTPUT_IDX_TO_WORD = {v:k for k,v in self.OUTPUT_WORD_TO_IDX.items()}
        
    
    def make_pytorch_datasets(self, data):
        data_set_train = HebrewWords(data.X_train, data.y_train, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
        data_set_val = HebrewWords(data.X_val, data.y_val, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
        data_set_test = HebrewWords(data.X_test, data.y_test, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
        return data_set_train, data_set_val, data_set_test
        
    def merge_datasets(self):
        data_merger_train = DataMerger(self.data_set_one.X_train, self.data_set_two.X_train, self.data_set_one.y_train, self.data_set_two.y_train)
        train_data_X, train_data_y = data_merger_train.merge_data()
        train_data_X, train_data_y = data_merger_train.shuffle_data(train_data_X, train_data_y)
        
        data_merger_val = DataMerger(self.data_set_one.X_val, self.data_set_two.X_val, self.data_set_one.y_val, self.data_set_two.y_val)
        val_data_X, val_data_y = data_merger_val.merge_data()
        val_data_X, val_data_y = data_merger_train.shuffle_data(val_data_X, val_data_y)
        return train_data_X, train_data_y, val_data_X, val_data_y
        
    def make_pytorch_merged_datasets(self, data_X_train, dat_y_train, data_X_val, data_y_val, data_X_test, data_y_test):
        data_set_train = HebrewWords(data_X_train, dat_y_train, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
        data_set_val = HebrewWords(data_X_val, data_y_val, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
        data_set_test = HebrewWords(data_X_test, data_y_test, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
        return data_set_train, data_set_val, data_set_test
        
    def make_data_loader(self, train_data, eval_data):
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, collate_fn=collate_transformer_fn)
        eval_dataloader = DataLoader(eval_data, batch_size=self.batch_size, collate_fn=collate_transformer_fn)
        return train_dataloader, eval_dataloader
        

class PipeLineTrain:
    def __init__(self,
                 input_file: str, 
                 output_file: str,
                 input_file2: str, 
                 output_file2: str,
                 INPUT_WORD_TO_IDX: dict, 
                 OUTPUT_WORD_TO_IDX: dict,
                 OUTPUT_IDX_TO_WORD: dict, 
                 nel: int,
                 ndl: int,
                 emb: int,
                 nh: int,
                 ffn: int,
                 dr: float,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float,
                 model_path: str,
                 evaluation_results_path: str,
                 epochs2: int,
                 beam_size: int,
                 beam_alpha: float,
                 length: int
                 ):
        self.input_file = input_file
        self.output_file = output_file
        self.input_file2 = input_file2
        self.output_file2 = output_file2
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX
        self.OUTPUT_IDX_TO_WORD = OUTPUT_IDX_TO_WORD
        self.nel = nel
        self.ndl = ndl
        self.emb = emb
        self.nh = nh
        self.ffn = ffn
        self.dr = dr
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.evaluation_results_path = evaluation_results_path
        self.epochs2 = epochs2
        self.beam_size = beam_size
        self.beam_alpha = beam_alpha
        self.length = length
        self.torch_seed = 42
        self.model_name = f'seq2seq_{self.length}seqlen_{self.learning_rate}lr_{self.epochs}_{self.epochs2}epochs_{self.emb}embedsize_{self.nh}nhead_{self.nel}nenclayers_{self.ndl}numdeclayers_transformer.pth'
        self.log_dir = f'runs/{self.input_file}_{self.output_file}/{self.length}seq_len_{self.learning_rate}lr_{self.epochs}_{self.epochs2}epochs_{self.emb}embsize_{self.nh}nhead_{self.nel}nenclayers_{self.ndl}numdeclayers_transformer'
        
        self.src_vocab_size = len(self.INPUT_WORD_TO_IDX)+2
        self.tgt_vocab_size = len(self.OUTPUT_WORD_TO_IDX)+2
        
    def initialize_model(self):
        transformer = initialize_transformer_model(self.nel, self.ndl, self.emb, self.nh, self.src_vocab_size, self.tgt_vocab_size, self.ffn, self.dr)
        return transformer
        
    def train_model(self, transformer, loss_fn, optimizer, train_dataloader, eval_dataloader, epochs, PAD_IDX):
        trained_model = train_transformer(transformer, loss_fn, optimizer, train_dataloader, 
                                          eval_dataloader, epochs, PAD_IDX, self.torch_seed, 
                                          self.learning_rate, self.log_dir, self.batch_size, self.INPUT_WORD_TO_IDX, 
                                          self.OUTPUT_WORD_TO_IDX)
        return trained_model        
        
    def save_model(self, trained_model, training_type):
        """
        Saves model and model configuration.
        The configuration consists of the sequence length and the dictionaries needed
        to convert input and output sequences from characters to integers.
        The configuration is needed if one wants to make predictions on new data.
        """
        model_config = {'seq_len': self.length,
                        'num_encoder_layers': self.nel,
                        'num_decoder_layers': self.ndl,
                        'emb_size': self.emb,
                        'nhead': self.nh,
                        'ffn_hid_dim': self.ffn,
                        'input_w2idx': self.INPUT_WORD_TO_IDX,
                        'output_w2idx': self.OUTPUT_WORD_TO_IDX,
                        'src_vocab_size': self.src_vocab_size,
                        'tgt_vocab_size': self.tgt_vocab_size
        }
        config_name = 'model_config' + self.model_name.rstrip('.pth') + '.json'
        
        model_folder = f'MODEL_{self.input_file}_{self.output_file}_{training_type}'
        if self.input_file2 and self.output_file2:
            model_folder = model_folder + f'_{self.input_file2}_{self.output_file2}'
        
        pth = os.path.join(self.model_path, model_folder)
        if not os.path.exists(pth):
            os.makedirs(pth)
        self.model_path_full = os.path.join(pth, self.model_name)
        with open(os.path.join(pth, config_name), 'w') as json_file:
            json.dump(model_config, json_file, indent=4)
        torch.save(trained_model.state_dict(), self.model_path_full)

        
    def evaluate_on_test_set(self, test_set, training_type):
    
        eval_path = f'{self.evaluation_results_path}/{self.input_file}_{self.output_file}_{training_type}'
        evaluation_file_name = f'{self.length}seq_len_{self.learning_rate}lr_{self.emb}embsize_{self.nh}nhead_transformer_{self.dr}dropout_{self.batch_size}_batchsize_{self.epochs}epochs_{self.beam_size}beamsize'
        if self.input_file2 and self.output_file2:
            eval_path = eval_path + f'_{self.input_file2}_{self.output_file2}'
            evaluation_file_name = evaluation_file_name + f'_epochs2_{self.epochs2}'

        evaluate_transformer_model(eval_path,
                                   evaluation_file_name,
                                   self.length,
                                   self.nel,
                                   self.ndl,
                                   self.emb,
                                   self.nh,
                                   self.src_vocab_size,
                                   self.tgt_vocab_size,
                                   self.ffn,
                                   self.model_path_full,
                                   test_set,
                                   self.OUTPUT_IDX_TO_WORD, 
                                   self.OUTPUT_WORD_TO_IDX,
                                   self.beam_size,
                                   self.beam_alpha)
