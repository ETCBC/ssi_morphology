import collections
import os

import torch

from config import device
from data import mc_expand
from model_transformer import Seq2SeqTransformer
from transformer_train_fns import generate_square_subsequent_mask
from wla_api import check_predictions


def greedy_decode(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int):
    """Function to generate output sequence using greedy algorithm """
    memory = (model.encode(src, src_mask)).to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len + 50):
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
    
def sequence_length_penalty(length: int, alpha: float=0.75) -> float:
    return ((5 + length) / (5 + 1)) ** alpha
    
def beam_search(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int, beam_size: int, alpha:int=0.75):
    """Function to generate output sequence using beam search algorithm.
    If the beam size is 0, greedy decoding will be applied.
    """
    #src, src_mask = src.to(device), src_mask.to(device)
    memory = (model.encode(src, src_mask)).to(device)

    #if not beam_size:
    #    return greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol)
    
    sequences = [[torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device), 0.0]]
    
    for i in range(max_len + 50):
        all_candidates = list()
        for j in range(len(sequences)):
            seq, score = sequences[j]
            
            tgt_mask = (generate_square_subsequent_mask(seq.size(0))
                    .type(torch.bool)).to(device)
            out = model.decode(seq, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob_beam = model.generator(out[:, -1])
            
            log_probs_all = torch.log_softmax(prob_beam, dim=1)
            log_probs_all = log_probs_all / sequence_length_penalty(i+1, alpha)
            prob_beam[0] = log_probs_all
            scores, indices = torch.topk(prob_beam, beam_size)

            for idx in range(beam_size):
                char_score = scores[0][idx].item()
                character_idx = indices[0][idx].item()
                
                if seq[-1].item() == end_symbol:
                    character_idcs = torch.cat([seq,
                        torch.ones(1, 1).type_as(src.data).fill_(end_symbol)], dim=0)
                    char_score = 0
                else:
                    character_idcs = torch.cat([seq,
                        torch.ones(1, 1).type_as(src.data).fill_(character_idx)], dim=0)
                candidate = [character_idcs, score + char_score]

                all_candidates.append(candidate)
        ordered_seqs_with_scores = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)[:beam_size]
               
        #ordered_seqs_with_scores = ordered[:beam_size]
        if all([seq[0][-1].item() == end_symbol for seq in ordered_seqs_with_scores]):
            break
    print('NEW')
    ordered_seqs_without_beam_scores = [seq[0] for seq in ordered_seqs_with_scores]
    print(ordered_seqs_without_beam_scores)
    return ordered_seqs_without_beam_scores


def num_to_char(output_idx_to_word_dict, tokens):
    character_string = ''.join([output_idx_to_word_dict[idx] for idx in list(tokens.cpu().numpy())]).replace('SOS', '').replace('EOS', '')
    return character_string


def translate(model: torch.nn.Module, encoded_sentence: str, OUTPUT_IDX_TO_WORD: dict, OUTPUT_WORD_TO_IDX: dict, beam_size: int, beam_alpha: float):
    model.eval()
    src = encoded_sentence.view(-1, 1).to(device)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)

    if not beam_size:
        tgt_tokens = greedy_decode(model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX['SOS'], OUTPUT_WORD_TO_IDX['EOS'])
        return num_to_char(OUTPUT_IDX_TO_WORD, tgt_tokens)
    
    tgt_tokens_list = beam_search(
        model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX['SOS'], OUTPUT_WORD_TO_IDX['EOS'], beam_size).flatten()
    
    # TODO: IMPLEMENT DECODE OF BEAM
    char_string_list = [num_to_char(OUTPUT_IDX_TO_WORD, numerical_seq) for numerical_seq in tgt_tokens_list]
    check_predictions()

    return character_string
    
    
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
                               OUTPUT_WORD_TO_IDX: dict,
                               beam_size: int,
                               beam_alpha: float):

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
            predicted = translate(loaded_transf.to(device), evaluation_data[i]['encoded_text'].to(device), OUTPUT_IDX_TO_WORD, OUTPUT_WORD_TO_IDX, beam_size, beam_alpha)
            true_val = evaluation_data[i]['output']

            predicted = mc_expand_whole_sequences(predicted)
            true_val = mc_expand_whole_sequences(true_val)
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

    
def mc_expand_whole_sequences(sequence):
    expanded_seq = ' '.join([mc_expand(word) for word in sequence.split()])
    return expanded_seq
