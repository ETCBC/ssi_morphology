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

from data import HebrewWords, DataReader, DataMerger, collate_fn, collate_transformer_fn, str2bool
from model_transformer import Seq2SeqTransformer
from model_rnn import HebrewEncoder, HebrewDecoder, save_encoder_decoder
from transformer_train_fns import initialize_transformer_model, train_transformer, evaluate
from rnn_train_fns import train_rnn
from evaluate_transformer import greedy_decode, translate, evaluate_transformer_model

from config import PAD_IDX, SOS_token, EOS_token



def main(PAD_IDX, SOS_token, EOS_token, args):
    """
    Train a bidirectional RNN or a Transformer model.
    
    Three different scenarios are possible, based on the use of the command line arguments. One uses
    -i, -o, and -ep                  -- The model is trained with one input (-i) and one output (-o) file and -ep epochs.
    -i, -o, -i2, -o2 and -ep         -- The model is trained with two input (-i, -i2) and two output (-o, -o2) files and -ep epochs. 
                                        The training and validation data are merged, and the test data come exclusively from -i2 and -o2.
    -i, -o, -i2, -o2 -ep , and -ep2  -- The model is trained with two input (-i, -i2) and two output (-o, -o2) files and -ep and -ep2 epochs. 
                                        The training and validation data of the two data sources are kept separately.
                                        The model is trained first on -i and -o with -ep epochs,
                                        and is trained further using -i2 and -o2 with ep2 epochs.
                                        The test data come exclusively from -i2 and -o2.
    
    Input: raw Hebrew text in ETCBC transcription.
    Output: Morphologically analyzed Hebrew text.
    Input and output consists of strings, therefore the models are seq2seq models.
    
    :i: filename of file with input sequences.
    :o: filename of file with output sequences.
    :l: number of graphical units in input sequence, e.g, "BR>CJT" has length 1, "BR>CJT BR>" has length 2, etc.
    :ep: number of training epochs
    :lr: learning rate
    :m: model type, is transformer or rnn
    :i2: second input file
    :o2: second output file
    :ep2: number of epochs for training second dataset
    
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="input_filename", help="Please specificy the input datafile in the folder data", type=str)
    parser.add_argument("-o", metavar="output_filename", help="Please specificy the output datafile in the folder data", type=str)
    parser.add_argument("-l", metavar="input_seq_len", help="Designate the number of words in one input string", type=int)
    parser.add_argument("-ep", metavar="epochs", help="Specify the number of training epochs", type=int)
    parser.add_argument("-lr", metavar="learning_rate", help="Specify the learning rate", type=float)
    
    parser.add_argument("-m", metavar="model_type", help="Chooses type of model: 'rnn' or 'transformer'", type=str)
    
    # Arguments for second (Syriac) dataset
    parser.add_argument("-i2", metavar="input_filename2", help="Please specificy the second input datafile in the folder data", type=str, default='', nargs='?')
    parser.add_argument("-o2", metavar="output_filename2", help="Please specificy the second output datafile in the folder data", type=str, default='', nargs='?')
    parser.add_argument("-ep2", metavar="epochs_syr", help="Specify the number of training epochs of syriac", type=int, default=0, nargs='?')
    
    # Hyperparameters of the Transformer model.
    parser.add_argument("-emb", metavar="emb_size", help="Optional: Embedding size, must be divisible by number of heads (nhead)", type=int, default=512, nargs='?')
    parser.add_argument("-nh", metavar="nhead", help="Optional: Number of heads", type=int, default=8, nargs='?')
    parser.add_argument("-nel", metavar="num_encoder_layers", help="Optional: Number of layers in encoder", type=int, default=3, nargs='?')
    parser.add_argument("-ndl", metavar="num_decoder_layers", help="Optional: Number of layers in decoder", type=int, default=3, nargs='?')
    parser.add_argument("-dr", metavar="dropout", help="Optional: dropout in transformer model", type=float, default=0.1, nargs='?')
    parser.add_argument("-b", metavar="batch_size", help="Optional: batch size during training", type=int, default=128, nargs='?')
    parser.add_argument("-wd", metavar="weight_decay", help="Optional: weight decay passed to optimizer", type=float, default=0.0, nargs='?')
    
    # Hyperparameters of the RNN model.
    parser.add_argument("-hd", metavar="hidden_dim", help="Optional: Number of cells in RNN layer", type=int, nargs='?')
    parser.add_argument("-nl", metavar="num_layers", help="Optional: Nmber of hidden layers", type=int, nargs='?')
    parser.add_argument("-bd", metavar="bidir", help="Optional: is the RNN model bidirectional or not", type=str2bool, const=True, default=False, nargs='?')
    
    args = parser.parse_args()
    
    batch_size = args.b
    val_plus_test_size = 0.3
    torch_seed = 42
    
    INPUT_WORD_TO_IDX = {
                         'PAD': PAD_IDX,
                         'SOS': SOS_token,
                         'EOS': EOS_token
                        }

    OUTPUT_WORD_TO_IDX = {
                          'PAD': PAD_IDX,
                          'SOS': SOS_token,
                          'EOS': EOS_token
                         }
    
    # load the dataset, and split 70/15/15 in train/val/test
    input_file = os.path.join("../data", args.i)
    output_file = os.path.join("../data", args.o)
    hebrew_data = DataReader(input_file, output_file, args.l, val_plus_test_size, INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX)
    
    # load the second dataset, and split 70/15/15 in test/val/test
    if args.i2 and args.o2:
        input_file_s = os.path.join("../data", args.i2)
        output_file_s = os.path.join("../data", args.o2)
        syriac_data = DataReader(input_file_s, output_file_s, args.l, val_plus_test_size, hebrew_data.INPUT_WORD_TO_IDX, hebrew_data.OUTPUT_WORD_TO_IDX)
    
        syriac_data.INPUT_IDX_TO_WORD = {v:k for k,v in syriac_data.INPUT_WORD_TO_IDX.items()}
        syriac_data.OUTPUT_IDX_TO_WORD = {v:k for k,v in syriac_data.OUTPUT_WORD_TO_IDX.items()}
        
        # Make Hebrew dicts identical to Syriac dicts.
        hebrew_data.INPUT_WORD_TO_IDX = syriac_data.INPUT_WORD_TO_IDX
        hebrew_data.OUTPUT_WORD_TO_IDX = syriac_data.OUTPUT_WORD_TO_IDX
        hebrew_data.INPUT_IDX_TO_WORD = syriac_data.INPUT_IDX_TO_WORD
        hebrew_data.OUTPUT_IDX_TO_WORD = syriac_data.OUTPUT_IDX_TO_WORD
        
        # Hebrew and Syriac together
        if not args.ep2:
            training_type = 'two_datasets_simultaneously'
            # Merge and shuffle Hebrew and Syriac data.
            data_merger_train = DataMerger(hebrew_data.X_train, syriac_data.X_train, hebrew_data.y_train, syriac_data.y_train)
            train_data_X, train_data_y = data_merger_train.merge_data()
            train_data_X, train_data_y = data_merger_train.shuffle_data(train_data_X, train_data_y)
    
            data_merger_val = DataMerger(hebrew_data.X_val, syriac_data.X_val, hebrew_data.y_val, syriac_data.y_val)
            val_data_X, val_data_y = data_merger_val.merge_data()
            val_data_X, val_data_y = data_merger_train.shuffle_data(val_data_X, val_data_y)

            # Make Pytorch datasets
            bible_train = HebrewWords(train_data_X, train_data_y, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
            bible_val = HebrewWords(val_data_X, val_data_y, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
            # Test data contain Syriac sequences only.
            bible_test = HebrewWords(syriac_data.X_test, syriac_data.y_test, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)

        # First Hebrew, then Syriac
        else:
            training_type = 'two_datasets_sequential'
            # Make Pytorch datasets
            bible_train = HebrewWords(hebrew_data.X_train, hebrew_data.y_train, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
            bible_val = HebrewWords(hebrew_data.X_val, hebrew_data.y_val, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
            bible_train_s = HebrewWords(syriac_data.X_train, syriac_data.y_train, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
            bible_val_s = HebrewWords(syriac_data.X_val, syriac_data.y_val, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
            # Test data contain Syriac sequences only.
            bible_test = HebrewWords(syriac_data.X_test, syriac_data.y_test, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
    
    # Only Hebrew data
    else:
        training_type = 'one_dataset'
        hebrew_data.INPUT_IDX_TO_WORD = {v:k for k,v in hebrew_data.INPUT_WORD_TO_IDX.items()}
        hebrew_data.OUTPUT_IDX_TO_WORD = {v:k for k,v in hebrew_data.OUTPUT_WORD_TO_IDX.items()}
        # Make Pytorch datasets
        bible_train = HebrewWords(hebrew_data.X_train, hebrew_data.y_train, hebrew_data.INPUT_WORD_TO_IDX, hebrew_data.OUTPUT_WORD_TO_IDX)
        bible_val = HebrewWords(hebrew_data.X_val, hebrew_data.y_val, hebrew_data.INPUT_WORD_TO_IDX, hebrew_data.OUTPUT_WORD_TO_IDX)
        bible_test = HebrewWords(hebrew_data.X_test, hebrew_data.y_test, hebrew_data.INPUT_WORD_TO_IDX, hebrew_data.OUTPUT_WORD_TO_IDX)
        
    src_vocab_size = len(hebrew_data.INPUT_WORD_TO_IDX)+2
    tgt_vocab_size = len(hebrew_data.OUTPUT_WORD_TO_IDX)+2
   
    if args.m == 'transformer':
    
        ffn_hid_dim = 512
        
        train_dataloader = DataLoader(bible_train, batch_size=batch_size, collate_fn=collate_transformer_fn)
        eval_dataloader = DataLoader(bible_val, batch_size=batch_size, shuffle=False, collate_fn=collate_transformer_fn)
        
        transformer = initialize_transformer_model(args.nel, 
                                                   args.ndl, 
                                                   args.emb, 
                                                   args.nh, 
                                                   src_vocab_size, 
                                                   tgt_vocab_size, 
                                                   ffn_hid_dim, args.dr)
                                         
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.wd)
        
        log_dir = f'runs/{args.i}_{args.o}/{args.l}seq_len_{torch_seed}seed_{args.lr}lr_{args.ep}epochs_{args.emb}embsize_{args.nh}nhead_{args.nel}nenclayers_{args.ndl}numdeclayers_transformer'
    
        trained_transformer = train_transformer(transformer, 
                                                loss_fn, 
                                                optimizer, 
                                                train_dataloader, 
                                                eval_dataloader, 
                                                args.ep, 
                                                PAD_IDX, 
                                                torch_seed, 
                                                args.lr, 
                                                log_dir, 
                                                batch_size, 
                                                hebrew_data.INPUT_WORD_TO_IDX, 
                                                hebrew_data.OUTPUT_WORD_TO_IDX)

        if args.i2 and args.o2 and args.ep2:
        
            train_dataloader_s = DataLoader(bible_train_s, batch_size=batch_size, collate_fn=collate_transformer_fn)
            eval_dataloader_s = DataLoader(bible_val_s, batch_size=50, shuffle=False, collate_fn=collate_transformer_fn)
           
            trained_transformer = train_transformer(trained_transformer, loss_fn, optimizer, train_dataloader_s, eval_dataloader_s, args.ep2, PAD_IDX, torch_seed, args.lr, log_dir, 
                   batch_size, syriac_data.INPUT_WORD_TO_IDX, syriac_data.OUTPUT_WORD_TO_IDX)
          
            model_path = './transformer_models'
            model_name = f'seq2seq_{args.l}seqlen_{torch_seed}seed_{args.lr}lr_{args.ep}epochs_{args.ep2}epochs2_{args.emb}embedsize_{args.nh}nhead_{args.nel}nenclayers_{args.ndl}numdeclayers_transformer.pth'
            torch.save(trained_transformer.state_dict(), os.path.join(model_path, model_name))
            evaluate_transformer_model(args.i, args.o, args.l, args.lr, args.ep, args.nel, 
                                   args.ndl, args.emb, args.nh,
                                   src_vocab_size, tgt_vocab_size, ffn_hid_dim,
                                   model_path, model_name, bible_test, args.dr, batch_size,
                                   syriac_data.OUTPUT_IDX_TO_WORD, syriac_data.OUTPUT_WORD_TO_IDX,
                                   input2=args.i2, output2=args.o2, epochs2=args.ep2, training_type=training_type)

        else:
            model_path = './transformer_models'
            model_name = f'seq2seq_{args.l}seqlen_{torch_seed}seed_{args.lr}lr_{args.ep}epochs_{args.emb}embedsize__{args.nh}nhead_{args.nel}nenclayers_{args.ndl}numdeclayers_transformer.pth'

            torch.save(trained_transformer.state_dict(), os.path.join(model_path, model_name))
        
            evaluate_transformer_model(args.i, args.o, args.l, args.lr, args.ep, args.nel, 
                                   args.ndl, args.emb, args.nh,
                                   src_vocab_size, tgt_vocab_size, ffn_hid_dim,
                                   model_path, model_name, bible_test, args.dr, batch_size,
                                   hebrew_data.OUTPUT_IDX_TO_WORD, hebrew_data.OUTPUT_WORD_TO_IDX)
                                                              
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
    main(PAD_IDX, SOS_token, EOS_token, sys.argv[1:])