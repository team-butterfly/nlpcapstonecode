from collections import namedtuple
import numpy as np

Batch = namedtuple("Batch", ["xs", "ys", "lengths"])

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

    def _next(self, size): # no shuffle
        idxs = (self._i + np.arange(size)) % self._limit
        next_i = (self._i + size) % self._limit

        self._is_new_epoch = next_i < self._i
        if self._is_new_epoch:
            self._epoch += 1
        self._i = next_i

        return Batch(self._batch.xs[idxs], self._batch.ys[idxs],
            self._batch.lengths[idxs] if self._batch.lengths is not None else None)

    def next(self, max_size):
        size = min(max_size, self._limit - self._i)
        idxs = np.arange(size) + self._i
        next_i = (self._i + size) % self._limit

        ret = Batch(self._batch.xs[idxs], self._batch.ys[idxs],
            self._batch.lengths[idxs] if self._batch.lengths is not None else None)

        self._is_new_epoch = next_i == 0
        if self._is_new_epoch:
            shuf = np.arange(self._limit)
            np.random.shuffle(shuf)
            self._batch = Batch(
                self._batch.xs[shuf],
                self._batch.ys[shuf],
                self._batch.lengths[shuf] if self._batch.lengths is not None else None)
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
