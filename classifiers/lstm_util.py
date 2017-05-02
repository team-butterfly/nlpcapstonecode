import numpy as np
import tensorflow as tf

from .batch import Batch
from utility import Emotion


NUM_LABELS  = len(Emotion)   # Number of labels for classifcation
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