from config import device

import torch

from transformer_train_fns import generate_square_subsequent_mask

# function to generate output sequence using greedy algorithm 
def greedy_decode(model: torch.nn.Module, src, src_mask, max_len: int, start_symbol: int, end_symbol: int):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
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
    #print('encoded_sentence: ', src)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, num_tokens-3, OUTPUT_WORD_TO_IDX["SOS"], OUTPUT_WORD_TO_IDX["EOS"]).flatten()
    print('target_tokens: ', tgt_tokens)
    
    return "".join([OUTPUT_IDX_TO_WORD[idx] for idx in list(tgt_tokens.cpu().numpy())]).replace("SOS", "").replace("EOS", "")