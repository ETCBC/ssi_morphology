import collections
import os

import torch

from config import device
from model_transformer import Seq2SeqTransformer
from transformer_train_fns import generate_square_subsequent_mask

# function to generate output sequence using greedy algorithm 
def greedy_decode(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1+20):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(device)
        print('memory', memory)
        print('mask', tgt_mask)
        out = model.decode(ys, memory, tgt_mask)
        print('out', out)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        print(prob)
        _, next_word = torch.max(prob, dim=1)
        print(next_word, next_word.item())
        print(torch.topk(prob, k=2, dim=1))
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                        
        if next_word == end_symbol:
            break
    return ys
    
#def beam_search(model, beam_width):
def beam_search(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int, beam_width: int):
    '''Decode an output from a model using beam search with specified beam_width'''
    
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)

    # beam_seq keeps track of words in each beam
    #beam_seq = np.empty((beam_width, 1), dtype=np.int32)
    ys = torch.ones(beam_width, 1).fill_(start_symbol).type(torch.long).to(device)

    # beam_log_probs is the likelihood of each beam
    beam_log_probs = np.zeros((beam_width,1))

    vocab_length = model.vocab_length
    prob_char_given_prev = np.empty((beam_width, vocab_length))

    done = False
    first_char = True
    while not done:

        #if first_char:
        #    prob_first_char = model.predict_next([])
        #    log_prob_first_char = np.log(prob_first_char)
        #    top_n, log_p = torch.topk(log_prob_first_char, beam_width, dim=1)
        #    beam_seq[:,0] = top_n[0]
        #    beam_log_probs[:,0] += log_p
        #    first_char = False
        #else:

        for beam in range(beam_width):
            prob_char_given_prev[beam] = model.predict_next(beam_seq[beam])
        log_prob_char_given_prev = np.log(prob_char_given_prev)
        log_prob_char = beam_log_probs + log_prob_char_given_prev
        top_n, log_p = get_top_n(log_prob_char, beam_width)
        beam_seq = torch.hstack((beam_seq[top_n[0]], top_n[1].reshape(-1,1)))

        beam_log_probs = log_p.reshape(-1,1)

        if len(beam_seq[0]) == max_len+10:
            done = True

    return beam_seq, beam_log_probs


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, encoded_sentence: str, OUTPUT_IDX_TO_WORD: dict, OUTPUT_WORD_TO_IDX: dict):
    model.eval()
    src = encoded_sentence.view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX['SOS'], OUTPUT_WORD_TO_IDX['EOS']).flatten()
    
    print(''.join([OUTPUT_IDX_TO_WORD[idx] for idx in list(tgt_tokens.cpu().numpy())]).replace('SOS', '').replace('EOS', ''))
    
    return ''.join([OUTPUT_IDX_TO_WORD[idx] for idx in list(tgt_tokens.cpu().numpy())]).replace('SOS', '').replace('EOS', '')
    
    
def evaluate_transformer_model(input_file, output_file, input_seq_len, lr, epochs, num_encoder_layers, num_decoder_layers, emb_size, 
                               nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim,
                               model_path, model_name, evaluation_data, dropout, batch_size,
                               OUTPUT_IDX_TO_WORD, OUTPUT_WORD_TO_IDX, training_type, **kwargs):

    loaded_transf = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size, 
                                       nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim)
                                 
    loaded_transf.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    loaded_transf.eval()

    word_eval_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    correct_complete_sequence = 0
    correct_all_words = [0 for i in range(input_seq_len)]
    
    eval_path = f'../evaluation_results_transformer/new_preprocessing_{input_file}_{output_file}_'
    evaluation_file_name = f'{input_seq_len}seq_len_{lr}lr_{emb_size}embsize_{nhead}nhead_transformer_{dropout}dropout_{batch_size}batchsize_epochs{epochs}_'
    
    eval_path = eval_path + f'{kwargs.get("input2", "_")}_{kwargs.get("output2", "_")}_{training_type}'
    evaluation_file_name = evaluation_file_name + f'epochs2_{kwargs.get("epochs2", "")}'
    
    isExist = os.path.exists(eval_path)
    if not isExist:
        os.makedirs(eval_path)
        
    
    
    # Aangepast voor Samaritaanse pentateuch
    with open(f'{eval_path}/results_{evaluation_file_name}.txt', 'w') as f:
        test_len = len(evaluation_data)
        for i in range(1): #(test_len):
    
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
