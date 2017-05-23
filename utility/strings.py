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
    """
    def __init__(self, word_iter, unk_threshold=1):
        """Create a StringStore from words in word_iter.
        If count(w) < unk_threshold, w is ignored.
        """
        counts = Counter(word_iter)
        vocab = [word for word, n in counts.items() if n >= unk_threshold]
        self._id2w = sorted(vocab)
        self._w2id = {word: idx for idx, word in enumerate(self._id2w)}
        assert len(self._id2w) == len(self._w2id)

    def __contains__(self, word):
        return word in self._w2id

    def __len__(self):
        """Return the vocabulary size."""
        return len(self._id2w)

    def vocab(self):
        """Return the set of known symbols."""
        return self._w2id.keys()

    def word2id(self, word):
        """Return an integer ID for word. Word must be in the vocabulary."""
        return self._w2id[word]

    def id2word(self, i):
        """Return the word with ID=i. Requires 0 <= i < len(me)."""
        return self._id2w[i]

    def count_vector(self, word_iter):
        """Return a vector v with length equal to my vocabulary size,
        where v[i] = count of instance of word with ID=i seen in word_iter.
        """
        as_ids = [self.word2id(word) for word in word_iter if word in self._w2id]
        return np.bincount(as_ids, minlength=len(self))

if __name__ == "__main__":
    inp = open("/homes/iws/brandv2/nlp/corpus/brown.txt").read().split()
    ss = StringStore(inp)
    for word in ss.vocab():
        assert ss.id2word(ss.word2id(word)) == word
