"""
    run the trained model using best first beam search
"""
from torch.utils.data import DataLoader, Subset
from .model import load_encoder_decoder
from .evaluate import score_beam_search, score
from .data import HebrewWords, collate_fn


if __name__ == "__main__":
    encoder, decoder = load_encoder_decoder(
           filename='runs/2layers_128hidden_42seed_0.001lr_bidir/model.27.pt'
           )

    encoder.eval()
    decoder.eval()

    dataset = HebrewWords('data/t-in_voc', 'data/t-out')

    loader = DataLoader(
            Subset(dataset, range(500)),
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
            )

    bs = score_beam_search(encoder, decoder, loader, max_length=20)
    print('Best first Beam search')
    print(bs)

    gs = score(encoder, decoder, loader, max_length=20)
    print('Greedy decoding')
    print(gs)
