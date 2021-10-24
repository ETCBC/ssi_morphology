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

from data import HebrewWords, collate_fn #, collate_transformer_fn
from model_transformer import Seq2SeqTransformer
from model_lstm import HebrewEncoder, HebrewDecoder, save_encoder_decoder #, reshape_hidden
from transformer_train_fns import initialize_transformer_model, train_transformer, evaluate
from lstm_train_fns import train_lstm
from evaluate_transformer import greedy_decode, translate


def main(args):
    """
    Train a bidirectional LSTM or a Transformer model.
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
    
    parser.add_argument("model_type", help="Chooses type of model: 'lstm' or 'transformer'", type=str)
    
    # Hyperparameters of the LSTM model.
    parser.add_argument("hidden_dim", help="Optional: Number of cells in LSTM layer", type=int, nargs='?')
    parser.add_argument("num_layers", help="Optional: Nmber of hidden layers", type=int, nargs='?')
    parser.add_argument("bidir", help="Optional: is the LSTM model bidirectional or not", type=bool, nargs='?')
    
    args = parser.parse_args()
    
    # load the dataset, and split 70/30 in test/eval
    input_file = os.path.join("../data", args.input_filename)
    output_file = os.path.join("../data", args.output_filename)
    bible = HebrewWords(input_file, output_file, args.input_seq_len)
    len_train = int(0.7 * len(bible))
    len_eval = len(bible) - len_train
    # alwyas use the same seed for train/test split
    training_data, evaluation_data = random_split(
            bible, [len_train, len_eval], generator=torch.Generator().manual_seed(42))
            
    torch_seed = 42
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    SRC_VOCAB_SIZE = len(bible.INPUT_WORD_TO_IDX)+2
    TGT_VOCAB_SIZE = len(bible.OUTPUT_WORD_TO_IDX)+2
    PAD_IDX = bible.INPUT_WORD_TO_IDX['PAD']
    #learning_rate = 0.0001
    #NUM_EPOCHS = 12
    
    # log settings
    #log_dir = f'runs/{args.learning_rate}lr_transf_runs/{num_layers}layers_{hidden_dim}hidden_{torch_seed}seed_{learning_rate}lr_'
    
    
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    eval_dataloader = DataLoader(evaluation_data, batch_size=50, shuffle=False, collate_fn=collate_fn)
    
    ##############################################################
    
    if args.model_type == 'transformer':
        transformer = initialize_transformer_model(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                         EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
                                         
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        
        log_dir = f'runs/{args.input_seq_len}seq_len_{torch_seed}seed_{args.learning_rate}lr_transformer'
    
        trained_transformer = train_transformer(transformer, loss_fn, optimizer, train_dataloader, eval_dataloader, args.epochs, PAD_IDX, torch_seed, learning_rate, log_dir, 
                   BATCH_SIZE, bible.INPUT_WORD_TO_IDX, bible.OUTPUT_WORD_TO_IDX) 
                   
    elif args.model_type == 'lstm':
        # create the network
        encoder = HebrewEncoder(input_dim=len(bible.INPUT_WORD_TO_IDX), hidden_dim=args.hidden_dim, num_layers=args.num_layers, bidir=args.bidir)
        decoder = HebrewDecoder(hidden_dim=1*args.hidden_dim, output_dim=len(bible.OUTPUT_WORD_TO_IDX), num_layers=args.num_layers)

        # function to optimize
        loss_function = nn.NLLLoss()

        # log settings
        log_dir = f'runs/{args.input_seq_len}seq_len_{args.num_layers}layers_{args.hidden_dim}hidden_{torch_seed}seed_{args.learning_rate}lr_lstm'

        # optimization strategy
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
        
        train_lstm(training_data=train_dataloader, evaluation_data=eval_dataloader,
          encoder=encoder, decoder=decoder,
          encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
          loss_function=loss_function, log_dir=log_dir,
          max_epoch=100, torch_seed=42, learning_rate=args.learning_rate, batch_size=20)
    

    ################################################################

    save_path = '.' 
    model_name = f'seq2seq_seqlen{args.input_seq_len}_epochs{NUM_EPOCHS}_transformer.pth'
    #model_name = 'seq2seq_seqlen4_epochs12_transformer19102021.pth'
    
    #torch.save(trained_transformer.state_dict(), os.path.join(save_path, model_name))

    loaded_transf = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
                                 
    loaded_transf.load_state_dict(torch.load(os.path.join(save_path, model_name)))
    loaded_transf.eval()


    word_eval_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    correct_complete_sequence = 0
    correct_all_words = [0 for i in range(args.input_seq_len)]

    test_len = 100
    print(bible.OUTPUT_IDX_TO_WORD)
    for i in range(test_len):
    
        predicted = translate(loaded_transf, evaluation_data[i]['encoded_text'], bible.OUTPUT_IDX_TO_WORD, bible.OUTPUT_WORD_TO_IDX)
        true_val = evaluation_data[i]['output']
        print(evaluation_data[i]['encoded_text'])
        print(predicted)
        print(true_val)
        predicted_words = predicted.split()
        true_val_words = true_val.split()
    
        #if len(predicted_words) != args.input_seq_len:
        #    continue
        
        if predicted == true_val:
            correct_complete_sequence += 1
        
        for word_idx in range(args.input_seq_len):
            try:
                if predicted_words[word_idx] == true_val_words[word_idx]:
                    correct_all_words[word_idx] += 1
            
                    word_eval_dict[true_val_words[word_idx]][word_idx].append('correct')
                else:
                    word_eval_dict[true_val_words[word_idx]][word_idx].append('wrong')
            except:
                continue

    print('complete string', correct_complete_sequence / test_len)
    print('distinct words', [correct_count / test_len for correct_count in correct_all_words])    

        
    


if __name__ == '__main__':
    main(sys.argv[1:])