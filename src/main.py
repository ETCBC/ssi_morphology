import sys
import argparse
import collections

import os.path
from timeit import default_timer as timer

from config import check_abort, abort_handler, device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler

from data import HebrewWords, collate_fn, collate_transformer_fn, str2bool
from model_transformer import Seq2SeqTransformer
from model_rnn import HebrewEncoder, HebrewDecoder, save_encoder_decoder
from transformer_train_fns import initialize_transformer_model, train_transformer, evaluate
from rnn_train_fns import train_rnn
from evaluate_transformer import greedy_decode, translate, evaluate_transformer_model


def main(args):
    """
    Train a bidirectional RNN or a Transformer model.
    Input: raw Hebrew text in ETCBC transcription.
    Output: Morphologically analyzed Hebrew text.
    Input and output consists of string, therefore the models are seq2seq models.   
    :input_filename: filename of file with input sequences.
    :output_filename: filename of file with output sequences.
    :seq_len: number of graphical units in input sequence, e.g, "BR>CJT" has length 1, "BR>CJT BR>" has length 2, etc.
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_filename", help="Please specificy the input datafile in the folder data", type=str)
    parser.add_argument("-o", metavar="output_filename", help="Please specificy the output datafile in the folder data", type=str)
    parser.add_argument("-l", metavar="input_seq_len", help="Designate the number of words in one input string", type=int)
    parser.add_argument("-ep", metavar="epochs", help="Specify the number of training epochs", type=int)
    parser.add_argument("-lr", metavar="learning_rate", help="Specify the learning rate", type=float)
    
    parser.add_argument("-m", metavar="model_type", help="Chooses type of model: 'rnn' or 'transformer'", type=str)
    
    # Hyperparameters of the Transformer model.
    parser.add_argument("-emb", metavar="emb_size", help="Optional: Embedding size, must be divisible by number of heads (nhead)", type=int, default=512, nargs='?')
    parser.add_argument("-nh", metavar="nhead", help="Optional: Number of heads", type=int, default=8, nargs='?')
    parser.add_argument("-nel", metavar="num_encoder_layers", help="Optional: Number of layers in encoder", type=int, default=3, nargs='?')
    parser.add_argument("-ndl", metavar="num_decoder_layers", help="Optional: Number of layers in decoder", type=int, default=3, nargs='?')
    parser.add_argument("-dr", metavar="dropout", help="Optional: dropout in transformer model", type=float, default=0.1, nargs='?')
    parser.add_argument("-b", metavar="batch_size", help="Optional: batch size during training", type=int, default=128, nargs='?')
    
    # Hyperparameters of the RNN model.
    parser.add_argument("-hd", metavar="hidden_dim", help="Optional: Number of cells in RNN layer", type=int, nargs='?')
    parser.add_argument("-nl", metavar="num_layers", help="Optional: Nmber of hidden layers", type=int, nargs='?')
    parser.add_argument("-bd", metavar="bidir", help="Optional: is the RNN model bidirectional or not", type=str2bool, const=True, default=False, nargs='?')
    
    args = parser.parse_args()
    
    batch_size = args.b
    test_size = 0.3
    
    # load the dataset, and split 70/15/15 in train/val/test
    input_file = os.path.join("../data", args.i)
    output_file = os.path.join("../data", args.o)
    bible = HebrewWords(input_file, output_file, args.l, test_size)
    len_train = int((1-test_size) * len(bible))

    torch_seed = 42

    # random_split() in PyTorch 1.3.1 does not have a parameter 'generator'
    torch.manual_seed(torch_seed)
    
    src_vocab_size = len(bible.INPUT_WORD_TO_IDX)+2
    tgt_vocab_size = len(bible.OUTPUT_WORD_TO_IDX)+2
    PAD_IDX = bible.INPUT_WORD_TO_IDX['PAD']
   
    if args.m == 'transformer':
    
        ffn_hid_dim = 512
        
        indices = list(range(len(bible)))
        
        upper_val_index = int((1 - 0.5*test_size) * len(bible))
        train_indices, val_indices, test_indices = indices[:len_train], indices[len_train:upper_val_index], indices[upper_val_index:]
        
        bible_train = Subset(bible, train_indices)
        bible_eval = Subset(bible, val_indices)
        bible_test = Subset(bible, test_indices)
        
        train_dataloader = DataLoader(bible_train, batch_size=batch_size, collate_fn=collate_transformer_fn)
        eval_dataloader = DataLoader(bible_eval, batch_size=50, shuffle=False, collate_fn=collate_transformer_fn)
        
        transformer = initialize_transformer_model(args.nel, args.ndl, 
                                                   args.emb, args.nh, src_vocab_size, tgt_vocab_size, ffn_hid_dim, args.dr)
                                         
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        
        log_dir = f'runs/{args.i}_{args.o}/{args.l}seq_len_{torch_seed}seed_{args.lr}lr_{args.ep}epochs_{args.emb}embsize_{args.nh}nhead_{args.nel}nenclayers_{args.ndl}numdeclayers_transformer'
    
        trained_transformer = train_transformer(transformer, loss_fn, optimizer, train_dataloader, eval_dataloader, args.ep, PAD_IDX, torch_seed, args.lr, log_dir, 
                   batch_size, bible.INPUT_WORD_TO_IDX, bible.OUTPUT_WORD_TO_IDX)

        model_path = './transformer_models'
        model_name = f'seq2seq_{args.l}seqlen_{torch_seed}seed_{args.lr}lr_{args.ep}epochs_{args.emb}embedsize__{args.nh}nhead_{args.nel}nenclayers_{args.ndl}numdeclayers_transformer.pth'    

        torch.save(trained_transformer.state_dict(), os.path.join(model_path, model_name))    
        
        evaluate_transformer_model(args.i, args.o, args.l, args.lr, args.ep, args.nel, 
                                   args.ndl, args.emb, args.nh,
                                  src_vocab_size, tgt_vocab_size, ffn_hid_dim,
                                   model_path, model_name, bible_test, 
                                   bible.OUTPUT_IDX_TO_WORD, bible.OUTPUT_WORD_TO_IDX)
                                                              
    elif args.m == 'rnn':
    
        train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
        eval_dataloader = DataLoader(evaluation_data, batch_size=50, shuffle=False, collate_fn=collate_fn)
        
        encoder = HebrewEncoder(input_dim=len(bible.INPUT_WORD_TO_IDX), hidden_dim=args.hd, num_layers=args.nl, bidir=args.bd)
        decoder = HebrewDecoder(hidden_dim=1*args.hd, output_dim=len(bible.OUTPUT_WORD_TO_IDX), num_layers=args.nl)

        
        loss_function = nn.NLLLoss()

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
        
        log_dir = f'runs/{args.l}seq_len_{args.nl}layers_{args.hd}hidden_{torch_seed}seed_{args.lr}lr_rnn'
        
        train_rnn(training_data=train_dataloader, evaluation_data=eval_dataloader,
          encoder=encoder, decoder=decoder,
          encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
          loss_function=loss_function, log_dir=log_dir,
          inp_wo2idx=bible.INPUT_WORD_TO_IDX, outp_w2idx=bible.OUTPUT_WORD_TO_IDX,
          max_epoch=args.ep, torch_seed=torch_seed, learning_rate=args.lr, batch_size=20
          )
    
        model_name = f'seq2seq_seqlen{args.l}_epochs{args.ep}_rnn.pth'


if __name__ == '__main__':
    main(sys.argv[1:])
