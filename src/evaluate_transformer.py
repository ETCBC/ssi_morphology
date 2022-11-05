import collections
import os

import torch

from config import device
from model_transformer import Seq2SeqTransformer
from transformer_train_fns import generate_square_subsequent_mask


def greedy_decode(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int):
    """Function to generate output sequence using greedy algorithm """
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1+20):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                        
        if next_word == end_symbol:
            break
    return ys
    
def beam_decode(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int):
    """Function to generate output sequence using greedy algorithm """
    print(src)
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    #print(memory)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    sequences_with_eos = []
    sequences = [[ys, 0.0]]
    
    for i in range(max_len-1+20):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])

        best_k2_probs, best_k2_idx = prob.topk(3)
        print('probs', best_k2_probs)
        scores = torch.log(best_k2_probs).view(3, -1)
        #log_probs = torch.log_softmax(best_k2_probs[:, -1], dim=1)
        log_probs = torch.log_softmax(best_k2_probs, dim=1)
        print('logprobs', log_probs)
        # SEE: https://kikaben.com/transformers-evaluation-details/
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for idx in range(3):
                char_score = scores[idx][0].item()
        #        character_idx = best_k2_idx[0][idx].item()
        #        character_idcs = torch.cat([seq,
        #                torch.ones(1, 1).type_as(src.data).fill_(character_idx)], dim=0)
        #        candidate = [character_idcs, score + char_score, character_idx]
        #        #print('CANDIDATE', candidate)
        #        all_candidates.append(candidate)
        #ordered = sorted(all_candidates, key=lambda tup:tup[1])
        
        #sequences = []
        #for cand in ordered:
        #   if cand[2] == end_symbol:
        #       sequences_with_eos.append((cand[0], cand[1]))
        #   else:
        #       sequences.append((cand[0], cand[1]))
               
        #sequences = ordered[:3]
        #print('SEQUENCES', sequences)
        
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        print('next word', next_word)
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        
                        
        if next_word == end_symbol:
            break
            
        print('ys', ys)
    return ys


def translate(model: torch.nn.Module, encoded_sentence: str, OUTPUT_IDX_TO_WORD: dict, OUTPUT_WORD_TO_IDX: dict):
    model.eval()
    src = encoded_sentence.view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_decode(
        model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX['SOS'], OUTPUT_WORD_TO_IDX['EOS']).flatten()
    
    return ''.join([OUTPUT_IDX_TO_WORD[idx] for idx in list(tgt_tokens.cpu().numpy())]).replace('SOS', '').replace('EOS', '')
    
    
def evaluate_transformer_model(eval_path: str, 
                               evaluation_file_name: str,
                               input_seq_len: int,
                               num_encoder_layers: int, 
                               num_decoder_layers: int, 
                               emb_size: int, 
                               nhead: int, 
                               src_vocab_size: int, 
                               tgt_vocab_size: int, 
                               ffn_hid_dim: int,
                               model_path_full: str, 
                               evaluation_data, 
                               OUTPUT_IDX_TO_WORD: dict, 
                               OUTPUT_WORD_TO_IDX: dict):

    loaded_transf = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size, 
                                       nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim)
                                 
    loaded_transf.load_state_dict(torch.load(model_path_full))
    loaded_transf.eval()

    word_eval_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    correct_complete_sequence = 0
    correct_all_words = [0 for i in range(input_seq_len)]

    if not os.path.exists(eval_path):
        os.makedirs(eval_path)

    with open(f'{eval_path}/results_{evaluation_file_name}.txt', 'w') as f:
        test_len = len(evaluation_data)
        for i in range(test_len):
            predicted = translate(loaded_transf.to(device), evaluation_data[i]['encoded_text'].to(device), OUTPUT_IDX_TO_WORD, OUTPUT_WORD_TO_IDX)
            true_val = evaluation_data[i]['output']
        
            f.write(f'Predicted {predicted}\n')
            f.write(f'Truevalue {true_val}\n')

            predicted_words = predicted.split()
            true_val_words = true_val.split()
            
            if predicted == true_val:
                correct_complete_sequence += 1
        
            for word_idx in range(input_seq_len):
                try:
                    if predicted_words[word_idx] == true_val_words[word_idx]:
                        correct_all_words[word_idx] += 1
            
                        word_eval_dict[true_val_words[word_idx]][word_idx].append('correct')
                    else:
                        word_eval_dict[true_val_words[word_idx]][word_idx].append('wrong')
                except:
                    continue
        f.write('\n')
        f.write(f'Correct complete strings {correct_complete_sequence / test_len}\n')
        f.write(f'Correct distinct words {[correct_count / test_len for correct_count in correct_all_words]}\n')
