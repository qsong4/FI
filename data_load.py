# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Note.
if safe, entities on the source side have the prefix 1, and the target side 2, for convenience.
For example, fpath1, fpath2 means source file path and target file path, respectively.
'''
import tensorflow as tf
from utils import calc_num_batches

def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>

    Returns
    two dictionaries.
    '''
    with open(vocab_fpath, 'r') as fr:
        vocab = [line.strip() for line in fr]

    #vocab = [line.split()[0] for line in open(vocab_fpath, 'r').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token, len(vocab)

def load_data(fpath, maxlen):
    '''Loads source and target data and filters out too lengthy samples.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    sents1, sents2 = [], []
    labels = []
    with open(fpath, 'r') as fr:
        for line in fr:
            content = line.strip().split("\t")
            sent1 = content[0]
            sent2 = content[1]
            label = int(content[2])
            if len(sent1) > maxlen:
                sent1 = sent1[len(sent1)-maxlen:]
            if len(sent2) > maxlen:
                sent2 = sent2[len(sent2)-maxlen:]
            sents1.append(sent1)
            sents2.append(sent2)
            labels.append(label)
    return sents1, sents2, labels


def encode(inp, dict):
    '''Converts string to number. Used for `generator_fn`.
    inp: 1d byte array.
    type: "x" (source side) or "y" (target side)
    dict: token2idx dictionary

    Returns
    list of numbers
    '''

    x = [dict.get(t, dict["<unk>"]) for t in inp]
    return x

def generator_fn(sents1, sents2, labels, vocab_fpath):
    '''Generates training / evaluation data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.

    yields
    xs: tuple of
        x: list of source token ids in a sent
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) sentence
    labels: tuple of
        decoder_input: decoder_input: list of encoded decoder inputs
        y: list of target token ids in a sent
        y_seqlen: int. sequence length of y
        sent2: str. target sentence
    '''
    token2idx, _, _ = load_vocab(vocab_fpath)
    for sent1, sent2, label in zip(sents1, sents2, labels):
        x = encode(sent1, token2idx)
        y = encode(sent2, token2idx)

        x_seqlen, y_seqlen = len(x), len(y)
        yield ((x, x_seqlen), (y, y_seqlen), (label))

def input_fn(sents1, sents2, labels, vocab_fpath, batch_size, shuffle=False):
    '''Batchify data
    sents1: list of source sents
    sents2: list of target sents
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)

    '''
    #((x, x_seqlen), (y, y_seqlen), (label))
    shapes = (([None], ()), ([None], ()), (()))
    types = ((tf.int32, tf.int32), (tf.int32, tf.int32), tf.int32)
    paddings = ((0, 0), (0, 0), 0)

    dataset = tf.data.Dataset.from_generator(
        generator_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sents1, sents2, labels, vocab_fpath))  # <- arguments for generator_fn. converted to np string arrays

    if shuffle: # for training
        dataset = dataset.shuffle(128*batch_size)

    dataset = dataset.repeat()  # iterate forever
    dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)

    return dataset

def get_batch(fpath, maxlen, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath: source file path. string.
    maxlen: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''
    sents1, sents2, labels = load_data(fpath, maxlen)
    batches = input_fn(sents1, sents2, labels, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)
