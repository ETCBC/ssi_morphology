"""Unit tests for data loading and conversion"""

from data import INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX
from data import encode_string, decode_string, mc_reduce, mc_expand
from data import HebrewVerses


def test_encode_decode_input():
    test = "B.:R;>CIJT B.@R@> >:ELOHIJM >;T HAC.@MAJIM W:>;T H@>@REY"
    enc_test = encode_string(test, INPUT_WORD_TO_IDX)
    dec_test = decode_string(enc_test, INPUT_WORD_TO_IDX)
    assert(dec_test == test)


def test_reduce_encode_decode_expand_output():
    test = "B-R>CJT/ BR>[ >LH(J(M/JM >T H-CMJ(M/(JM W->T H->RY/:a"
    reduced = mc_reduce(test)
    enc_test = encode_string(reduced, OUTPUT_WORD_TO_IDX)
    dec_test = decode_string(enc_test, OUTPUT_WORD_TO_IDX)
    expanded = mc_expand(dec_test)
    assert(expanded == test)


def test_hebrewbible():
    bible = HebrewVerses('data/t-in_voc', 'data/t-out')
    verse = bible[0]
    assert(verse['book'] == 'Gen')
    assert(verse['chapter'] == 1)
    assert(verse['verse'] == 1)
    assert(verse['text'] == "B.:R;>CIJT B.@R@> >:ELOHIJM >;T HAC.@MAJIM W:>;T H@>@REY")
