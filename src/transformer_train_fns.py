from config import device
from data import decode_string
from signal import signal, SIGINT, SIG_DFL
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model_transformer import Seq2SeqTransformer

from config import check_abort, abort_handler

def initialize_transformer_model(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM):

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(device)

    return transformer


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, PAD_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_transformer(model, loss_fn, optimizer, train_dataloader, eval_dataloader, NUM_EPOCHS, PAD_IDX, torch_seed, learning_rate, log_dir, batch_size, INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX):

    torch.manual_seed(torch_seed)
    
    # Tell Python to run the abort_handler() function when SIGINT is recieved
    signal(SIGINT, abort_handler)
    
    # tensorboard
    writer = SummaryWriter(log_dir)
    
    timer_start = time.time()
    timer = timer_start
    counter = 0
    
    model.train()
    losses = 0

    for epoch in range(NUM_EPOCHS):
        if check_abort():
            break
   
        for src, encoder_lengths, tgt, decoder_target, decoder_lengths in train_dataloader:
            if check_abort():
                break
    
            counter += batch_size
            print(counter)
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            loss.backward()

            optimizer.step()
            losses += loss.item()  

            # every N steps, print some diagnostics
            if counter % 10*batch_size == 0:
                oldtimer = timer
                timer = time.time()
                
                model.eval()
                
                sentence = src[:, 0].view(-1)  # take the first sentence
                sentence = sentence[0:encoder_lengths[0]]  # trim padding
                sentence = decode_string(sentence, INPUT_WORD_TO_IDX) 

                gold = tgt[:, 0].view(-1)  # take the first sentence
                gold = gold[1:decoder_lengths[0]]  # trim padding and SOS
                gold = decode_string(gold, OUTPUT_WORD_TO_IDX)                
        
                # to add: system
                #sentence = src[:, 0].view(-1)
                #print(sentence)
                
                #print(f'step={counter} epoch={epoch} t={timer - timer_start} dt={timer - oldtimer} batchloss={loss.item()}')
                
                #print(f'\tverse: {sentence}\tgold: {gold}\t') # system: {system}')
                #writer.add_text('sample', sentence + "<=>" + system, global_step=counter)
                #writer.add_scalar('Loss/train', loss.item(), global_step=counter)
                
                model.train()
                
    return model

    
def evaluate(model, loss_fn, val_dataloader, PAD_IDX):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)