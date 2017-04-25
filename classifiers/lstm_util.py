import numpy as np
import tensorflow as tf

from .batch import Batch


NUM_LABELS  = 5   # Number of labels for classifcation
HIDDEN_SIZE = 64  # Size of LSTM hidden state
NUM_LAYERS  = 1   # Number of stacked LSTM layers
VOCAB_SIZE  = 256 # Number of symbols recognized in input to the LSTM. (256 ASCII chars)
SEQ_LEN     = 160 # Max sequence length supported by LSTM


def encode_raw_inputs(raw_inputs):
    """Convert list of strings to array of arrays of ASCII values"""
    max_seen = max(len(st) for st in raw_inputs)
    if max_seen > SEQ_LEN:
        raise ValueError("Input exceeds max sequence length of {}".format(SEQ_LEN))
    
    def to_array(sentence):
        return np.array([ord(char) for char in sentence], dtype=np.int32)

    return np.array([to_array(sentence) for sentence in raw_inputs])


def make_batch(char_inputs, labels):
    """
    For the Lstm, a Batch should have:
        `xs` an array with dims [size, LEN] where LEN<=seq_len, where each row
            is a sequence that will be input to the model, consisting of an array of
            ASCII values padded with zeros to reach LEN.
        `ys` the correct output labels.
        `lengths` the true lengths of each input sequence (so that the LSTM can ignore the padding)
    """
    size = len(char_inputs)
    max_length = max(len(sent) for sent in char_inputs)

    x = np.zeros([size, max_length], dtype=np.int32)
    y = np.array(labels, dtype=np.int) if labels is not None else None
    lengths = np.zeros([size], dtype=np.int)
    
    for i, sent in enumerate(char_inputs):
        pad = max_length - len(sent)
        x[i] = np.pad(sent, (0, pad), mode="constant")
        lengths[i] = len(sent)

    return Batch(xs=x, ys=y, lengths=lengths)


class Minibatcher():
    """
    Minibatcher is essentially an iterator that cycles through subsets of
    a larger Batch object that contains a full data set.
    It also records its progress through the full data set to help monitor
    training.
    """
    def __init__(self, full_batch):
        self._batch = full_batch
        self._i = 0
        self._epoch = 1
        self._limit = len(self._batch.xs)
        self._is_new_epoch = False

    def next(self, size):
        idxs = (self._i + np.arange(size)) % self._limit
        next_i = (self._i + size) % self._limit

        self._is_new_epoch = next_i < self._i
        if self._is_new_epoch:
            self._epoch += 1
        self._i = next_i

        return Batch(self._batch.xs[idxs], self._batch.ys[idxs], self._batch.lengths[idxs])
    
    @property
    def epoch_progress(self):
        return self._i / self._limit

    @property
    def cur_epoch(self):
        return self._epoch

    @property
    def is_new_epoch(self):
        return self._is_new_epoch
