''' Handling the data io '''
import os
import argparse
import logging
import pdb

import dill as pickle
import urllib
from tqdm import tqdm
import sys
import codecs
import spacy
import torch
import tarfile
import torchtext.data
import torchtext.datasets
from torchtext.datasets import TranslationDataset
from torchtext.data import TabularDataset
import transformer.Constants as Constants
from learn_bpe import learn_bpe
from apply_bpe import BPE
import sentencepiece as spm
import pdb


def main_wo_bpe():
    '''
    Usage: python preprocess.py -lang_src de -lang_trg en -save_data multi30k_de_en.pkl -share_vocab
    '''

    spacy_support_langs = ['de_core_news_sm', 'el', 'en_core_web_sm', 'es', 'fr', 'it', 'lt', 'nb', 'nl', 'pt']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lang_src', choices=spacy_support_langs)
    parser.add_argument('-lang_trg', choices=spacy_support_langs)
    parser.add_argument('-data_dir', required=True)
    parser.add_argument('-train_path', type=str, default="train.csv")
    parser.add_argument('-valid_path', type=str, default="valid.csv")
    parser.add_argument('-test_path', type=str, default="test.csv")
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-data_src', type=str, default=None)
    parser.add_argument('-data_trg', type=str, default=None)
    parser.add_argument('-tokenizer', type=str, choices=['spacy', 'spm'])
    parser.add_argument('-t_type', type=str, choices=['unigram', 'bpe'])
    parser.add_argument('-data_type', type=str, choices=['naver', 'hate', 'bias'])

    parser.add_argument('-max_len', type=int, default=100)
    parser.add_argument('-min_word_count', type=int, default=3)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    #parser.add_argument('-ratio', '--train_valid_test_ratio', type=int, nargs=3, metavar=(8,1,1))
    #parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    # assert not any([opt.data_src, opt.data_trg]), 'Custom data input is not support now.'
    # assert not any([opt.data_src, opt.data_trg]) or all([opt.data_src, opt.data_trg])
    print(opt)

    if opt.tokenizer == 'spacy':
        src_lang_model = spacy.load(opt.lang_src)
        trg_lang_model = spacy.load(opt.lang_trg)

        def tokenize_src(text):
            return [tok.text for tok in src_lang_model.tokenizer(text)]

        def tokenize_trg(text):
            return [tok.text for tok in trg_lang_model.tokenizer(text)]

    elif opt.tokenizer == 'spm':
        if opt.t_type == "unigram":
            src_sp = spm.SentencePieceProcessor()
            trg_sp = spm.SentencePieceProcessor()
            spm_dir = "data/tokenizer"

            src_sp.Load(os.path.join(spm_dir, f"{opt.data_type}_{opt.t_type}.model"))
            trg_sp.Load(os.path.join(spm_dir, f"{opt.data_type}_{opt.t_type}.model"))
        elif opt.t_type == 'bpe':
            src_sp = spm.SentencePieceProcessor()
            trg_sp = spm.SentencePieceProcessor()
            spm_dir = "data/tokenizer"

            src_sp.Load(os.path.join(spm_dir, "bpe", f"{opt.data_type}_{opt.t_type}.model"))
            trg_sp.Load(os.path.join(spm_dir, "bpe", f"{opt.data_type}_{opt.t_type}.model"))

        def tokenize_src(text):
            return [tok for tok in src_sp.EncodeAsPieces(text)]

        def tokenize_trg(text):
            return [tok for tok in trg_sp.EncodeAsPieces(text)]

    SRC = torchtext.data.Field(
        tokenize=tokenize_src, lower=not opt.keep_case,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    TRG = torchtext.data.Field(
        tokenize=tokenize_trg, lower=not opt.keep_case,
        pad_token=Constants.PAD_WORD, init_token=Constants.BOS_WORD, eos_token=Constants.EOS_WORD)

    MAX_LEN = opt.max_len
    MIN_FREQ = opt.min_word_count

    # if not all([opt.data_src, opt.data_trg]):
    #     assert {opt.lang_src, opt.lang_trg} == {'de_core_news_sm', 'en_core_web_sm'}
    # else:
    #     # Pack custom txt file into example datasets
    #     raise NotImplementedError

    def filter_examples_with_length(x):
        return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN


    # train = TabularDataset(path=os.path.join(opt.data_dir, opt.train_path), format="csv",
    #                        fields=[("src", SRC), ("trg", TRG)])
    # val = TabularDataset(path=os.path.join(opt.data_dir, opt.valid_path), format="csv",
    #                      fields=[("src", SRC), ("trg", TRG)])
    # test = TabularDataset(path=os.path.join(opt.data_dir, opt.test_path), format="csv",
    #                       fields=[("src", SRC), ("trg", TRG)])

    train = TabularDataset(path=os.path.join(opt.data_dir, opt.train_path), format="csv",
                           fields=[("src", SRC), ("trg", TRG)])
    val = TabularDataset(path=os.path.join(opt.data_dir, opt.valid_path), format="csv",
                         fields=[("src", SRC), ("trg", TRG)])
    test = TabularDataset(path=os.path.join(opt.data_dir, opt.test_path), format="csv",
                          fields=[("src", SRC), ("trg", TRG)])



    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    print('[Info] Get source language vocabulary size:', len(SRC.vocab))
    TRG.build_vocab(train.trg, min_freq=MIN_FREQ)
    print('[Info] Get target language vocabulary size:', len(TRG.vocab))

    # if opt.share_vocab:
    #     print('[Info] Merging two vocabulary ...')
    #     for w, _ in SRC.vocab.stoi.items():
    #         # TODO: Also update the `freq`, although it is not likely to be used.
    #         if w not in TRG.vocab.stoi:
    #             TRG.vocab.stoi[w] = len(TRG.vocab.stoi)
    #     TRG.vocab.itos = [None] * len(TRG.vocab.stoi)
    #     for w, i in TRG.vocab.stoi.items():
    #         TRG.vocab.itos[i] = w
    #     SRC.vocab.stoi = TRG.vocab.stoi
    #     SRC.vocab.itos = TRG.vocab.itos
    #     print('[Info] Get merged vocabulary size:', len(TRG.vocab))


    data = {
        'settings': opt,
        'vocab': {'src': SRC, 'trg': TRG},
        'train': train.examples,
        'valid': val.examples,
        'test': test.examples}
    pdb.set_trace()

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    pickle.dump(data, open(opt.save_data, 'wb'))


if __name__ == '__main__':
    main_wo_bpe()
    #main()