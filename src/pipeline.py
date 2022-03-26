import os

import torch
from torch.utils.data import DataLoader

from data import collate_transformer_fn, DataMerger, DataReader, HebrewWords
from transformer_train_fns import initialize_transformer_model, train_transformer
from evaluate_transformer import evaluate_transformer_model

class PipeLine:
    def __init__(self, 
                 input_file: str, 
                 output_file: str, 
                 length: int, 
                 INPUT_WORD_TO_IDX dict, 
                 OUTPUT_WORD_TO_IDX: dict, 
                 nel: int,
                 ndl: int,
                 emb: int,
                 nh: int,
                 ffn: int,
                 dr: float,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float,
                 input_file2=None: str, 
                 output_file2=None: str,
                 epochs2=0: int):
                 
        self.input_file = input_file
        self.output_file = output_file
        self.length = length
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX
        self.nel = nel
        self.ndl = ndl
        self.emb = emb
        self.nh = nh
        self.ffn = ffn
        self.dr = dr
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_plus_test_size = 0.3
        self.learning_rate = learning_rate
        self.input_file2 = input_file2
        self.output_file2 = output_file2
        self.epochs2 = epochs2
        self.torch_seed = 42
        self.model_path = '../transformer_models'
        self.model_name = f'seq2seq_{self.length}seqlen_{self.learning_rate}lr_{self.epochs}_{self.epochs2}epochs_{self.emb}embedsize_{self.nh}nhead_{self.nel}nenclayers_{self.ndl}numdeclayers_transformer.pth'
        self.log_dir = f'runs/{self.input_file}_{self.output_file}/{self.length}seq_len_{self.learning_rate}lr_{self.epochs}_{self.epochs2}epochs_{self.emb}embsize_{self.nh}nhead_{self.nel}nenclayers_{self.ndl}numdeclayers_transformer'
        
        self.data_set_one = DataReader(input_file, output_file, length, self.val_plus_test_size, INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX)
        
        if input_file2 and output_file2:
            self.data_set_two = DataReader(input_file2, output_file2, length, self.val_plus_test_size, self.INPUT_WORD_TO_IDX, self.OUTPUT_WORD_TO_IDX)
            self.INPUT_WORD_TO_IDX = self.data_set_two.INPUT_WORD_TO_IDX
            self.OUTPUT_WORD_TO_IDX = self.data_set_two.OUTPUT_WORD_TO_IDX
            
        self.INPUT_IDX_TO_WORD = {v:k for k,v in self.INPUT_WORD_TO_IDX.items()}
        self.OUTPUT_IDX_TO_WORD = {v:k for k,v in self.OUTPUT_WORD_TO_IDX.items()}
        
        self.src_vocab_size = len(self.INPUT_WORD_TO_IDX)+2
        self.tgt_vocab_size = len(self.OUTPUT_WORD_TO_IDX)+2
    
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
        
    def initialize_model(self):
        transformer = initialize_transformer_model(self.nel, self.ndl, self.emb, self.nh, self.src_vocab_size, self.tgt_vocab_size, self.ffn, self.dr)
        return transformer
        
    def train_model(self, transformer, loss_fn, optimizer, train_dataloader, eval_dataloader, PAD_IDX):
        trained_model = train_transformer(transformer, loss_fn, optimizer, train_dataloader, 
                                          eval_dataloader, self.epochs, PAD_IDX, self.torch_seed, 
                                          self.learning_rate, self.log_dir, self.batch_size, self.INPUT_WORD_TO_IDX, 
                                          self.OUTPUT_WORD_TO_IDX)
        return trained_model
        
    def save_model(self, trained_model):
        torch.save(trained_model.state_dict(), os.path.join(self.model_path, self.model_name))
        
    def evaluate_on_test_set(self, test_set, training_type):
        evaluate_transformer_model(self.input_file, self.output_file, self.length, self.learning_rate, 
                                   self.epochs, self.nel, self.ndl, self.emb, self.nh,
                                   self.src_vocab_size, self.tgt_vocab_size, self.ffn,
                                   self.model_path, self.model_name, test_set, self.dr, self.batch_size,
                                   self.OUTPUT_IDX_TO_WORD, self.OUTPUT_WORD_TO_IDX, training_type,
                                   input2=self.input_file2, output2=self.output_file2, epochs2=self.epochs2)
                                   
    
    
def str2bool(v: str) -> bool:
    """
    Helper function needed to be able to use 
    boolean variables in the command line arguments.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
