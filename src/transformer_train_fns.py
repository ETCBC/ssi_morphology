import time

from Levenshtein import distance
from numpy import array
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from config import check_abort, abort_handler
from config import device
from data import decode_string
from model_transformer import Seq2SeqTransformer
from signal import signal, SIGINT, SIG_DFL

def initialize_transformer_model(num_encoder_layers, num_decoder_layers, emb_size, nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim, dropout):

    transformer = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size, 
                                 nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim, dropout)
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
    
class TrainingSession():
   def __init__(self, model, loss_function):
      self.model = model
      self.loss_function = loss_function
      self.n_epochs = 1
      self.optimizer = None
      self.trainloader = None
      self.validloader = None
      self.PAD_IDX = None
      

def edit_distance(t1, t2):
    o = ord(' ')
    s1 = ''.join([chr(o + i) for i in t1])
    s2 = ''.join([chr(o + i) for i in t2])
    return distance(s1, s2)


def accuracy(target, labels):
    tgt_out = labels[1:, :]
    values, predictions = torch.max(target, -1)
    wrong = 0
    for i in range(len(tgt_out)):
        wrong += edit_distance(tgt_out[i], predictions[i])
    return 1 - (wrong / array(tgt_out.size()).prod()).item()


def criterion(f, target, labels):
   tgt_out = labels[1:, :]
   return f(target.reshape(-1, target.shape[-1]), tgt_out.reshape(-1))


def run_model(session, src, tgt):
   tgt_input = tgt[:-1, :]
   src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
      create_mask(src, tgt_input, session.PAD_IDX)
   return session.model(src, tgt_input, src_mask, tgt_mask,
      src_padding_mask, tgt_padding_mask, src_padding_mask)


def train_step(session):
   train_loss = 0.0
   train_accy = 0.0
   n = 0
   for src, tgt in session.trainloader:
      src = src.to(device)
      tgt = tgt.to(device)
      session.optimizer.zero_grad()
      target = run_model(session, src, tgt)
      loss = criterion(session.loss_function, target, tgt)
      loss.backward()
      session.optimizer.step()
      delta = len(src)
      train_loss += loss.item() * delta
      train_accy += accuracy(target, tgt) * delta
      n += delta
   return train_loss / n, train_accy / n


def valid_step(session):
   valid_loss = 0.0
   valid_accy = 0.0
   n = 0
   for src, tgt in session.validloader:
      src = src.to(device)
      tgt = tgt.to(device)
      target = run_model(session, src, tgt)
      loss = criterion(session.loss_function, target, tgt)
      delta = len(src)
      valid_loss += loss.item() * delta
      valid_accy += accuracy(target, tgt) * delta
      n += delta
   return valid_loss / n, valid_accy / n


def run_training(session, writer):
    print('Epoch', 'Training loss', 'Training accuracy',
                'Validation loss', 'Validation accuracy', sep='\t')
    for epoch in range(session.n_epochs):
        if check_abort():
            break
        session.model.train()
        train_loss, train_accy = train_step(session)
        session.model.eval()
        valid_loss, valid_accy = valid_step(session)
        print(epoch+1, train_loss, train_accy,
                     valid_loss, valid_accy, sep='\t')
        writer.add_scalar("train loss", train_loss, epoch)
        writer.add_scalar("validation loss", valid_loss, epoch)
        writer.add_scalar("train accuracy", train_accy, epoch)
        writer.add_scalar("validation accuracy", valid_accy, epoch)
    return valid_loss


def train_transformer(model, loss_fn, optimizer, train_dataloader, eval_dataloader, num_epochs, PAD_IDX, torch_seed, learning_rate, log_dir, batch_size, INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX):

    torch.manual_seed(torch_seed)
    
    # Tell Python to run the abort_handler() function when SIGINT is recieved
    signal(SIGINT, abort_handler)
    
    # tensorboard
    writer = SummaryWriter(log_dir)

    session = TrainingSession(model, loss_fn)
    session.n_epochs = num_epochs
    session.optimizer = optimizer
    session.trainloader = train_dataloader
    session.validloader = eval_dataloader
    session.PAD_IDX = PAD_IDX
    val_loss = run_training(session, writer)
    writer.add_hparams({
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "loss_function": type(loss_fn).__name__,
        "optimizer_encoder": type(optimizer).__name__,
    }, {
        "hparam/validation loss": val_loss
    })
    writer.close()
    return model
