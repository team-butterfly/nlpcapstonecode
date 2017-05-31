"""
Utilities for classifiers 
"""
from pprint import pformat
from os.path import join as pjoin
from os import mkdir
from shutil import rmtree
import numpy as np
import tensorflow as tf
from utility import console, Emotion


def pad_to_max_len(xs):
    maxlen = max(len(x) for x in xs)
    ret = np.zeros([len(xs), maxlen], dtype=xs.dtype)
    for i, x in enumerate(xs):
        ret[i, :len(x)] = x
    return ret


def lengths(xs):
    return np.fromiter(map(len, xs), dtype=np.int, count=len(xs))


def xavier(size_in, size_out):
    d = np.sqrt(6 / (size_in + size_out))
    return tf.random_uniform((size_in, size_out), minval=-d, maxval=d)


class TrainingSession():

    def __init__(self, model_name, run_name):
        """
        Prepares a training session by making the log/checkpoint directories:
        * Log dir:        log/<model-name>/<run-name>
        * Checkpoint dir: ckpts/<model-name>/<run-name>
        * Checkpoint file ckpts/<model-name>/<run-name>/<run-name>
        """
        try:
            self.logdir = pjoin("log", model_name, run_name)
            self.ckptdir = pjoin("ckpts", model_name, run_name)
            self.ckpt_file = pjoin(self.ckptdir, run_name)
            console.log("Making logdir", self.logdir)
            console.log("Making checkpoint dir", self.ckptdir)
            mkdir(self.logdir)
            mkdir(self.ckptdir)
        except FileExistsError:
            console.warn("Logging or checkpoint directory already exists; "
                         + "choose a unique name for this training instance.")
            raise

    def erase_files(self):
        console.log("Removing logdir", self.logdir)
        console.log("Removing checkpoint dir", self.ckptdir)
        rmtree(self.logdir)
        rmtree(self.ckptdir)


class HParams():
    """
    Hyperparameters 'struct'
    """
    def __init__(self):
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.hidden_size = 200
        self.keep_prob_in = 0.5
        self.keep_prob_out = 0.5
        self.batch_size = 128
        self.vocab_size = 20000
        self.embed_size = 200
 
    def __str__(self):
        return pformat(self.__dict__) 


class Batch():

    def __init__(self, xs, ys, lengths):
        assert isinstance(xs, np.ndarray)
        assert isinstance(ys, np.ndarray) or ys is None
        assert isinstance(lengths, np.ndarray) or lengths is None
        self.xs = xs
        self.ys = ys
        self.lengths = lengths


class Minibatcher():
    """
    Minibatcher is essentially an iterator that cycles through subsets of
    a larger Batch object that contains a full data set.
    It shuffles the data set on each new epoch.
    It also records its progress through the full data set to help monitor
    training.
    """
    def __init__(self, full_batch, seed=5):
        assert isinstance(full_batch, Batch)
        self._batch = full_batch
        self._i = 0
        self._epoch = 1
        self._limit = len(self._batch.xs)
        self._is_new_epoch = False
        self._rand = np.random.RandomState(seed)

    def __len__(self):
        return len(self._batch.xs)

    def _shuffle(self):
        shuf = np.arange(self._limit)
        self._rand.shuffle(shuf)
        self._batch = Batch(
            self._batch.xs[shuf],
            self._batch.ys[shuf],
            self._batch.lengths[shuf] if self._batch.lengths is not None else None)

    def next(self, max_size, pad_per_batch=False):
        size = min(max_size, self._limit - self._i)
        idxs = np.arange(size) + self._i
        next_i = (self._i + size) % self._limit

        xs = self._batch.xs[idxs]
        if pad_per_batch:
            xs = pad_to_max_len(xs)

        ret = Batch(
            xs,
            self._batch.ys[idxs],
            self._batch.lengths[idxs] if self._batch.lengths is not None else None)

        self._is_new_epoch = next_i == 0
        if self._is_new_epoch:
            self._shuffle()
            self._epoch += 1
        self._i = next_i

        return ret


    @property
    def epoch_progress(self):
        return self._i / self._limit

    @property
    def cur_epoch(self):
        return self._epoch

    @property
    def is_new_epoch(self):
        return self._is_new_epoch
