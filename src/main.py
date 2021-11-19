import sys
import argparse
import collections

import os.path
from timeit import default_timer as timer

from config import check_abort, abort_handler, device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from data import HebrewWords, collate_fn, collate_transformer_fn, str2bool
from model_transformer import Seq2SeqTransformer
from model_rnn import HebrewEncoder, HebrewDecoder, save_encoder_decoder #, reshape_hidden
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
    parser.add_argument("input_filename", help="Please specificy the input datafile in the folder data", type=str)
    parser.add_argument("output_filename", help="Please specificy the output datafile in the folder data", type=str)
    parser.add_argument("input_seq_len", help="Designate the number of words in one input string", type=int)
    parser.add_argument("epochs", help="Specify the number of training epochs", type=int)
    parser.add_argument("learning_rate", help="Specify the learning rate", type=float)
    
    parser.add_argument("model_type", help="Chooses type of model: 'rnn' or 'transformer'", type=str)
    
    # Hyperparameters of the Transformer model.
    parser.add_argument("emb_size", help="Optional: Embedding size, must be divisible by number of heads (nhead)", type=int, default=512, nargs='?')
    parser.add_argument("nhead", help="Optional: Number of heads", type=int, default=8, nargs='?')
    parser.add_argument("num_encoder_layers", help="Optional: Number of layers in encoder", type=int, default=3, nargs='?')
    parser.add_argument("num_decoder_layers", help="Optional: Number of layers in decoder", type=int, default=3, nargs='?')
    
    
    # Hyperparameters of the RNN model.
    parser.add_argument("hidden_dim", help="Optional: Number of cells in RNN layer", type=int, nargs='?')
    parser.add_argument("num_layers", help="Optional: Nmber of hidden layers", type=int, nargs='?')
    parser.add_argument("bidir", help="Optional: is the RNN model bidirectional or not", type=str2bool, const=True, default=False, nargs='?')
    
    args = parser.parse_args()
    
    
    # load the dataset, and split 70/30 in test/eval
    input_file = os.path.join("../data", args.input_filename)
    output_file = os.path.join("../data", args.output_filename)
    bible = HebrewWords(input_file, output_file, args.input_seq_len)
    len_train = int(0.7 * len(bible))
    len_eval = len(bible) - len_train

    torch_seed = 42
    batch_size = 128

    # random_split() in PyTorch 1.3.1 does not have a parameter 'generator'
    # generator=torch.Generator().manual_seed(torch_seed)
    torch.manual_seed(torch_seed)
    training_data, evaluation_data = random_split(bible, [len_train, len_eval])

    src_vocab_size = len(bible.INPUT_WORD_TO_IDX)+2
    tgt_vocab_size = len(bible.OUTPUT_WORD_TO_IDX)+2
    PAD_IDX = bible.INPUT_WORD_TO_IDX['PAD']
 
    
    if args.model_type == 'transformer':
    
        ffn_hid_dim = 512
        train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_transformer_fn)
        eval_dataloader = DataLoader(evaluation_data, batch_size=50, shuffle=False, collate_fn=collate_transformer_fn)
        
        transformer = initialize_transformer_model(args.num_encoder_layers, args.num_decoder_layers, 
                                                   args.emb_size, args.nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim)
                                         
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        
        log_dir = f'runs/{args.input_seq_len}seq_len_{torch_seed}seed_{args.learning_rate}lr_{args.epochs}epochs_{args.emb_size}embsize_{args.nhead}nhead_{args.num_encoder_layers}nenclayers_{args.num_decoder_layers}numdeclayers_transformer'
    
        trained_transformer = train_transformer(transformer, loss_fn, optimizer, train_dataloader, eval_dataloader, args.epochs, PAD_IDX, torch_seed, args.learning_rate, log_dir, 
                   batch_size, bible.INPUT_WORD_TO_IDX, bible.OUTPUT_WORD_TO_IDX)

        model_path = './transformer_models'
        model_name = f'seq2seq_{args.input_seq_len}seqlen_{torch_seed}seed_{args.learning_rate}lr_{args.epochs}epochs_{args.emb_size}embedsize__{args.nhead}nhead_{args.num_encoder_layers}nenclayers_{args.num_decoder_layers}numdeclayers_transformer.pth'    

        torch.save(trained_transformer.state_dict(), os.path.join(model_path, model_name))    

        evaluate_transformer_model(args.input_seq_len, args.learning_rate, args.epochs, args.num_encoder_layers, 
                                   args.num_decoder_layers, args.emb_size, args.nhead,
                                   src_vocab_size, tgt_vocab_size, ffn_hid_dim,
                                   model_path, model_name, evaluation_data, 
                                   bible.OUTPUT_IDX_TO_WORD, bible.OUTPUT_WORD_TO_IDX)
                                                              
    elif args.model_type == 'rnn':
    
        train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
        eval_dataloader = DataLoader(evaluation_data, batch_size=50, shuffle=False, collate_fn=collate_fn)
        
        encoder = HebrewEncoder(input_dim=len(bible.INPUT_WORD_TO_IDX), hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidir=args.bidir)
        decoder = HebrewDecoder(hidden_dim=1*args.hidden_dim, output_dim=len(bible.OUTPUT_WORD_TO_IDX), num_layers=args.num_layers)

        
        loss_function = nn.NLLLoss()

        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
        
        log_dir = f'runs/{args.input_seq_len}seq_len_{args.num_layers}layers_{args.hidden_dim}hidden_{torch_seed}seed_{args.learning_rate}lr_rnn'
        
        train_rnn(training_data=train_dataloader, evaluation_data=eval_dataloader,
          encoder=encoder, decoder=decoder,
          encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
          loss_function=loss_function, log_dir=log_dir,
          inp_wo2idx=bible.INPUT_WORD_TO_IDX, outp_w2idx=bible.OUTPUT_WORD_TO_IDX,
          max_epoch=args.epochs, torch_seed=torch_seed, learning_rate=args.learning_rate, batch_size=20
          )
    
        model_name = f'seq2seq_seqlen{args.input_seq_len}_epochs{args.epochs}_rnn.pth'


if __name__ == '__main__':
    main(sys.argv[1:])
