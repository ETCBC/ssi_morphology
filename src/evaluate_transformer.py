import collections
import os
from typing import List

import torch

from config import device, wla_url
from data import mc_expand
from model_transformer import Seq2SeqTransformer
from transformer_train_fns import generate_square_subsequent_mask
from wla_api import check_predictions


def greedy_decode(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int):
    """Function to generate output sequence using greedy algorithm 
    
    """
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
    It returns a list with beam_size vectors, each representing
    """
    memory = (model.encode(src, src_mask)).to(device)
    
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
                char_score = scores[0][idx].detach().item()
                character_idx = indices[0][idx].detach().item()
                
                if seq[-1].item() == end_symbol:
                    character_idcs = torch.cat([seq,
                        torch.ones(1, 1).type_as(src.data).fill_(end_symbol)], dim=0)
                    char_score = 0
                else:
                    character_idcs = torch.cat([seq,
                        torch.ones(1, 1).type_as(src.data).fill_(character_idx)], dim=0)
                candidate = [character_idcs, score + char_score]
                all_candidates.append(candidate)
       
        ordered_seqs_with_scores = (sorted(all_candidates, key=lambda tup:tup[1], reverse=True))
        sequences = ordered_seqs_with_scores[:beam_size]
               
        if all([seq[0][-1].item() == end_symbol for seq in sequences]):
            break

    ordered_seqs_without_beam_scores = [seq[0].flatten() for seq in sequences]
    return ordered_seqs_without_beam_scores


def num_to_char(output_idx_to_word_dict: dict, tokens) -> str:
    character_string = ''.join([output_idx_to_word_dict[idx] for idx in list(tokens.cpu().numpy())]).replace('SOS', '').replace('EOS', '')
    return character_string


def process_predicted_results(predicted_strings_list: List[str], 
                              language: str, 
                              version: str) -> str:
    splitted_preds = [pred.split() for pred in predicted_strings_list]
    best_words = []
    for same_word_predictions in zip(*splitted_preds):
        same_word_predictions = list(set(same_word_predictions))
        best_prediction = check_predictions(wla_url, language, version, same_word_predictions)
        best_words.append(best_prediction)
    best_sequence = ' '.join(best_words)
    return best_sequence


def translate(model: torch.nn.Module, 
              encoded_sentence: str, 
              OUTPUT_IDX_TO_WORD: dict, 
              OUTPUT_WORD_TO_IDX: dict, 
              beam_size: int, 
              beam_alpha: float,
              language: str=None,
              version: str=None):
    """
    Makes predictions and decodes predictions to text, using greedy decoding or beam search.
    This is done on a sequence of words.
    If language and version are defined, an API call is made to the ETCBC server to check idiomatic correctness of the words in a sequence.
    The beam search potentially produces beam_size different words. They are all checked, and the first idiomatically correct word is returned.
    If there is no idiomatically correct word available, the best predicted word is returned. The API call is only done with new predictions, not during model evaluation on the test set.

    Output:
    best_sequence: str  Complete sequence of words.
    """
    model.eval()
    src = encoded_sentence.view(-1, 1).to(device)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(device)

    if not beam_size:
        tgt_tokens_list = [greedy_decode(model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX['SOS'], OUTPUT_WORD_TO_IDX['EOS'])]
    
    tgt_tokens_list = beam_search(
        model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX['SOS'], OUTPUT_WORD_TO_IDX['EOS'], beam_size, beam_alpha)
    
    predicted_strings_list = [num_to_char(OUTPUT_IDX_TO_WORD, numerical_seq) for numerical_seq in tgt_tokens_list]
    predicted_strings_list = [mc_expand_whole_sequences(predicted) for predicted in predicted_strings_list]
    if language and version:
        best_sequence = process_predicted_results(predicted_strings_list, language, version)
    else:
        best_sequence = predicted_strings_list[0]
    return best_sequence
    
    
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
                                 
    loaded_transf.load_state_dict(torch.load(model_path_full, map_location=device))
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

            #predicted = mc_expand_whole_sequences(predicted)
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

    
def mc_expand_whole_sequences(sequence: str) -> str:
    expanded_seq = ' '.join([mc_expand(word) for word in sequence.split()])
    return expanded_seq
