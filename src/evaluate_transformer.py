import collections
import os
from config import device
import torch
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


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, encoded_sentence: str, OUTPUT_IDX_TO_WORD: dict, OUTPUT_WORD_TO_IDX: dict):
    model.eval()
    src = encoded_sentence.view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, num_tokens, OUTPUT_WORD_TO_IDX["SOS"], OUTPUT_WORD_TO_IDX["EOS"]).flatten()
    
    return "".join([OUTPUT_IDX_TO_WORD[idx] for idx in list(tgt_tokens.cpu().numpy())]).replace("SOS", "").replace("EOS", "")
    
    
def evaluate_transformer_model(input_file, output_file, input_seq_len, lr, epochs, num_encoder_layers, num_decoder_layers, emb_size, 
                               nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim,
                               model_path, model_name, evaluation_data, OUTPUT_IDX_TO_WORD, 
                               OUTPUT_WORD_TO_IDX):

    loaded_transf = Seq2SeqTransformer(num_encoder_layers, num_decoder_layers, emb_size, 
                                       nhead, src_vocab_size, tgt_vocab_size, ffn_hid_dim)
                                 
    loaded_transf.load_state_dict(torch.load(os.path.join(model_path, model_name)))
    loaded_transf.eval()

    word_eval_dict = collections.defaultdict(lambda: collections.defaultdict(list))

    correct_complete_sequence = 0
    correct_all_words = [0 for i in range(input_seq_len)]
    
    eval_path = f'./evaluation_results_transformer/new_preprocessing_heb_syr_mixed_{input_file}_{output_file}'
    isExist = os.path.exists(eval_path)
    if not isExist: 
        os.makedirs(eval_path)
        
    evaluation_file_name = f'{input_seq_len}seq_len_{lr}lr_epochs{epochs}_{emb_size}embsize_{nhead}nhead_transformer'
    with open(f'{eval_path}/results_{evaluation_file_name}.txt', 'w') as f:
            
        test_len = len(evaluation_data)
        for i in range(test_len):
    
            predicted = translate(loaded_transf.to(device), evaluation_data[i]['encoded_text'].to(device), OUTPUT_IDX_TO_WORD, OUTPUT_WORD_TO_IDX)
            true_val = evaluation_data[i]['output']
        
            f.write(f'Predicted {predicted}\n')
            f.write(f'Truevalue {true_val}\n')

            predicted_words = predicted.split()
            true_val_words = true_val.split()
            
            #if len(predicted_words) != input_seq_len:
            #    continue
            
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
        f.write(f'Correct distinct words {[correct_count / test_len for correct_count in correct_all_words]}')
