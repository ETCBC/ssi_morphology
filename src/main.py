import sys
import argparse

from config import check_abort, abort_handler, device

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from enums import TrainingType, Language, WordGrammarVersion
from pipeline import PipeLineTrain
from pipeline_predict import PipeLinePredict
from utils import str2bool

from config import PAD_IDX, SOS_token, EOS_token, MODEL_PATH, EVALUATION_RESULTS_PATH


def main(args):
    """
    Train a Transformer model or make predictions on new data with a trained model.
    
    Three different scenarios are possible, based on the use of the command line arguments. One uses
    -i, -o, and -ep                  -- The model is trained with one input (-i) and one output (-o) file and -ep epochs.
    -i, -o, -i2, -o2 and -ep         -- The model is trained with two input (-i, -i2) and two output (-o, -o2) files and -ep epochs. 
                                        The training and validation data are merged, and the test data come exclusively from -i2 and -o2.
    -i, -o, -i2, -o2 -ep , and -ep2  -- The model is trained with two input (-i, -i2) and two output (-o, -o2) files and -ep and -ep2 epochs. 
                                        The training and validation data of the two data sources are kept separately.
                                        The model is trained first on -i and -o with -ep epochs,
                                        and is trained further using -i2 and -o2 with ep2 epochs.
                                        The test data come exclusively from -i2 and -o2.
    
    Input: raw Hebrew and/or Syriac text in ETCBC transcription.
    Output: Morphologically analyzed Hebrew or Syriac text.
    Input and output consists of strings, therefore the models are seq2seq models.
    
    Required arguments:
    :i: filename of file with input sequences.
    :o: filename of file with output sequences.
    :l: number of graphical units in input sequence, e.g, "BR>CJT" has length 1, "BR>CJT BR>" has length 2, etc.
    :ep: number of training epochs
    :lr: learning rate
    
    E.g., run:
    python main.py -i=t-in_voc -o=t-out -l=4 -ep=3 -lr=0.0001
    
    :i2: second input file
    :o2: second output file
    :ep2: number of epochs for training second dataset
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", metavar="mode", help="Is the mode training a new model or predicting with trained model. Can be 'train' or 'predict", type=str)
    
    # Argument required for predict mode.
    parser.add_argument("-pcf", metavar="predict_config_file", help="Name of yaml file for predictions", type=str, nargs='?')
    
    # Arguments required for training mode.
    parser.add_argument("-i", metavar="input_filename", help="Please specificy the input datafile in the folder data", type=str, nargs='?')
    parser.add_argument("-o", metavar="output_filename", help="Please specificy the output datafile in the folder data", type=str, nargs='?')
    parser.add_argument("-l", metavar="input_seq_len", help="Designate the number of words in one input string", type=int, nargs='?')
    parser.add_argument("-ep", metavar="epochs", help="Specify the number of training epochs", type=int, nargs='?')
    parser.add_argument("-lr", metavar="learning_rate", help="Specify the learning rate", type=float, nargs='?')
    
    # Arguments for training second (Syriac) dataset.
    parser.add_argument("-i2", metavar="input_filename2", help="Please specificy the second input datafile in the folder data", type=str, default='', nargs='?')
    parser.add_argument("-o2", metavar="output_filename2", help="Please specificy the second output datafile in the folder data", type=str, default='', nargs='?')
    parser.add_argument("-ep2", metavar="epochs_syr", help="Specify the number of training epochs of syriac", type=int, default=0, nargs='?')
    
    # Hyperparameters of the Transformer model.
    parser.add_argument("-emb", metavar="emb_size", help="Optional: Embedding size, must be divisible by number of heads (nhead)", type=int, default=512, nargs='?')
    parser.add_argument("-nh", metavar="nhead", help="Optional: Number of heads", type=int, default=8, nargs='?')
    parser.add_argument("-nel", metavar="num_encoder_layers", help="Optional: Number of layers in encoder", type=int, default=3, nargs='?')
    parser.add_argument("-ffn", metavar="ffn_hid_dim", help="Optional: Feed Forward Network Hidden Dimension", type=int, default=512, nargs='?')
    parser.add_argument("-ndl", metavar="num_decoder_layers", help="Optional: Number of layers in decoder", type=int, default=3, nargs='?')
    parser.add_argument("-dr", metavar="dropout", help="Optional: dropout in transformer model", type=float, default=0.1, nargs='?')
    parser.add_argument("-b", metavar="batch_size", help="Optional: batch size during training", type=int, default=128, nargs='?')
    parser.add_argument("-wd", metavar="weight_decay", help="Optional: weight decay passed to optimizer", type=float, default=0.0, nargs='?')
    
    # Evaluate on test set after training the model.
    parser.add_argument("-et", metavar="eval_test", help="Optional: evaluate at the end on test set (True) or not (False)", type=str2bool, const=True, default=False, nargs='?')
    parser.add_argument("-sz", metavar="beam_size", help="Optional: size of beam during beam size decoding. If 0 is chosen, decoding takes place with greedy decoding", type=int, default=3, nargs='?')
    parser.add_argument("-ba", metavar="beam_alpha", help="Optional: alpha value regulates the penalty for longer sequences during beam search.", type=float, default=0.75, nargs='?')

    args = parser.parse_args()
    
    assert args.mo
    
    if args.mo == 'predict':
        assert args.pcf
    elif args.mo == 'train':
        assert args.i
        assert args.o
        assert args.l
        assert args.ep
        assert args.lr
    
    if args.mo == 'predict':
        pipeline_predict = PipeLinePredict(args.pcf)
    
    elif args.mo == 'train':
        if args.i2 and args.o2 and args.ep2:
            training_type = TrainingType.TWO_DATASETS_SEQUENTIALLY
        elif args.i2 and args.o2:
            training_type = TrainingType.TWO_DATASETS_SIMULTANEOUSLY
        elif args.i and args.o:
            training_type = TrainingType.ONE_DATASET
        
        assert training_type

    
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
    
        pipeline = PipeLineTrain(args.i, 
                        args.o, 
                        args.l, 
                        INPUT_WORD_TO_IDX, 
                        OUTPUT_WORD_TO_IDX, 
                        args.nel, 
                        args.ndl, 
                        args.emb, 
                        args.nh, 
                        args.ffn, 
                        args.dr, 
                        args.b, 
                        args.ep, 
                        args.lr,  
                        MODEL_PATH,
                        EVALUATION_RESULTS_PATH,
                        args.i2,
                        args.o2, 
                        args.ep2,
                        args.sz,
                        args.ba,
                        val_plus_test_size=0.3
                        )

        if training_type == TrainingType.TWO_DATASETS_SIMULTANEOUSLY:
            train_data_X, train_data_y, val_data_X, val_data_y = pipeline.merge_datasets()
            hebrew_train, hebrew_val, test_set = pipeline.make_pytorch_merged_datasets(train_data_X, 
                                                                                   train_data_y, 
                                                                                   val_data_X, 
                                                                                   val_data_y, 
                                                                                   pipeline.data_set_two.X_test, 
                                                                                   pipeline.data_set_two.y_test)
        elif training_type == TrainingType.TWO_DATASETS_SEQUENTIALLY:
            hebrew_train, hebrew_val, _ = pipeline.make_pytorch_datasets(pipeline.data_set_one)
            syr_train, syr_val, test_set = pipeline.make_pytorch_datasets(pipeline.data_set_two)
        elif training_type == TrainingType.ONE_DATASET:
            hebrew_train, hebrew_val, test_set = pipeline.make_pytorch_datasets(pipeline.data_set_one)

        # Train model on (first) dataset
        train_dataloader, eval_dataloader = pipeline.make_data_loader(hebrew_train, hebrew_val)
        transformer = pipeline.initialize_model()
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        optimizer = optim.Adam(transformer.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.wd)
        trained_transformer = pipeline.train_model(transformer, loss_fn, optimizer, train_dataloader, eval_dataloader, args.ep, PAD_IDX)
    
        # Save and evaluate model
        if training_type in {TrainingType.TWO_DATASETS_SIMULTANEOUSLY, TrainingType.ONE_DATASET}:
            pipeline.save_model(trained_transformer, training_type.name)
            if args.et:
                pipeline.evaluate_on_test_set(test_set, training_type.name)
         
        # Train model on second dataset, save model and evaluate it.
        elif training_type == TrainingType.TWO_DATASETS_SEQUENTIALLY:
            train_dataloader_s, eval_dataloader_s = pipeline.make_data_loader(syr_train, syr_val)
            trained_transformer = pipeline.train_model(trained_transformer, loss_fn, optimizer, train_dataloader_s, eval_dataloader_s, args.ep2, PAD_IDX)
            pipeline.save_model(trained_transformer, training_type.name)
            if args.et:
                pipeline.evaluate_on_test_set(test_set, training_type.name)


if __name__ == '__main__':
    main(sys.argv[1:])
    