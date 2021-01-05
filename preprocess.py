import argparse
import torch
from TextProcessor import TextProcessor, TextIterator, TextDataset
from onmt_modules.onmt.bin.build_vocab import build_vocab_main
from onmt_modules.onmt.utils.parse import ArgumentParser
from onmt_modules.onmt.opts import _add_dynamic_corpus_opts, \
    _add_dynamic_fields_opts, _add_dynamic_transform_opts, config_opts, _add_reproducibility_opts

import configargparse

def build_save_text_dataset(src_corpus, tgt_corpus, fields,
                                      corpus_type, opt):


    src_seq_length_trunc = 0
    tgt_seq_length_trunc = 0
    src_seq_length = 1000
    tgt_seq_length = 1000

    src_iter = TextIterator(src_corpus, src_seq_length_trunc,
        "src")
    
    tgt_iter = TextIterator(
        tgt_corpus, tgt_seq_length_trunc,
        "tgt",
        assoc_iter=src_iter)
    ret_list = []
    dataset = TextDataset(
            fields, src_iter, tgt_iter,
            src_iter.num_feats, tgt_iter.num_feats,
            src_seq_length=src_seq_length,
            tgt_seq_length=tgt_seq_length,
            dynamic_dict=True)

    dataset.fields = []

    pt_file = "{:s}.{:s}.{:d}.pt".format(
            opt.save_data, corpus_type, 1)
    print(" * saving %s data shard to %s." % (corpus_type, pt_file))
    print(dataset)
    torch.save(dataset, pt_file)

    ret_list.append(pt_file)

def build_save_dataset(corpus_type, fields, opt):
    if corpus_type == 'train':
        src_corpus = opt.train_src
        tgt_corpus = opt.train_tgt
    else:
        src_corpus = opt.valid_src
        tgt_corpus = opt.valid_tgt

    return build_save_text_dataset(
            src_corpus, tgt_corpus, fields,
            corpus_type, opt)
    #return

def build_save_vocab(train_dataset, fields, opts):
    src_words_min_frequency = 0
    tgt_words_min_frequency = 0
    src_vocab_size = 50000
    tgt_vocab_size = 50000
    build_vocab_main(opts)

    # Can't save fields, so remove/reconstruct at training time.
    #vocab_file = opt.save_data + '.vocab.pt'
    #torch.save(onmt.io.save_fields_to_vocab(fields), vocab_file)
    return

def dynamic_prepare_opts(parser, build_vocab_only=False):
    """Options related to data prepare in dynamic mode.

    Add all dynamic data prepare related options to parser.
    If `build_vocab_only` set to True, then only contains options that
    will be used in `onmt/bin/build_vocab.py`.
    """
    config_opts(parser)
    parser.add_argument('-train_src', type = str)
    parser.add_argument('-train_tgt', type = str)
    parser.add_argument('-data_type', type = str)
    parser.add_argument('-valid_src', type = str)
    parser.add_argument('-valid_tgt', type = str)

    #parser.add_argument('-src_vocab', type = str)
    #parser.add_argument('-tgt_vocab', type = str)
    #parser.add_argument('-share_vocab', type = str)

    _add_dynamic_corpus_opts(parser, build_vocab_only=build_vocab_only)
    _add_dynamic_fields_opts(parser, build_vocab_only=build_vocab_only)
    _add_dynamic_transform_opts(parser)

    if build_vocab_only:
        _add_reproducibility_opts(parser)
        # as for False, this will be added in _add_train_general_opts



def preprocess():
    '''parser = argparse.ArgumentParser(
        description='preprocess.py')
    
    parser.add_argument('-train_src', type = str)
    parser.add_argument('-train_tgt', type = str)
    parser.add_argument('-data_type', type = str)
    parser.add_argument('-save_data', type = str)
    parser.add_argument('-valid_src', type = str)
    parser.add_argument('-valid_tgt', type = str)

    parser.add_argument('-src_vocab', type = str)
    parser.add_argument('-tgt_vocab', type = str)
    parser.add_argument('-share_vocab', type = str)


    opt = parser.parse_args()'''

    parser = ArgumentParser(description='build_vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only=True)
    opts, unknown = parser.parse_known_args()

    train_src = opts.train_src 
    train_tgt = opts.train_tgt 
    num_sorce_features = TextProcessor().get_num_features(train_src, 'src')
    num_target_features = TextProcessor().get_num_features(train_tgt, 'tgt')
    print('# source features', num_sorce_features)
    print('# target features', num_target_features)

    fields = TextProcessor().get_fields(num_sorce_features, num_target_features)

    train_dataset_files = build_save_dataset('train', fields, opts)

    build_save_vocab(train_dataset_files, fields, opts)

    print("Building & saving validation data...")
    build_save_dataset('valid', fields, opts)

if __name__ == "__main__":
    preprocess()