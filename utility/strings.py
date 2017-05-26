"""
Defines some useful operations for strings/vocabularies
"""
from enum import Enum
from collections import Counter
import numpy as np

def read_vocab(fn):
    with open(fn, "r") as f:
        lines = f.readlines()
        index_word = np.empty(len(lines), dtype="<U64")
        word_index = {}
        for i, line in enumerate(lines):
            word = line[:-1]
            index_word[i] = word
            word_index[word] = i
        return index_word, word_index

class StringStore():
    """
    StringStore remembers integer IDs for each unique string in an iterable.
    Ignores case.
    """
    def __init__(self, word_iter, limit):
        counts = Counter(map(lambda s: s.lower(), word_iter))
        self._id2w = [word for word, _ in counts.most_common(limit-1)]
        self._id2w.append("<unk>")
        self._w2id = {word: idx for idx, word in enumerate(self._id2w)}
        assert len(self._id2w) == len(self._w2id)

    def __contains__(self, word):
        return word in self._w2id

    def __len__(self):
        """Return the vocabulary size."""
        return len(self._id2w)

    def vocab(self):
        """Return a list of known symbols."""
        return self._id2w.copy()

    def word2id(self, word):
        """Return an integer ID for word. Word may be unk."""
        if word not in self._w2id:
            word = "<unk>"
        return self._w2id[word]

    def id2word(self, i):
        """Return the word with ID=i. Requires 0 <= i < len(me)."""
        return self._id2w[i]

    def count_vector(self, word_iter):
        """Return a vector v with length equal to my vocabulary size,
        where v[i] = count of word with ID=i seen in word_iter.
        """
        return np.bincount([self.word2id(w) for w in word_iter], minlength=len(self))
