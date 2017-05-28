"""
Defines some useful operations for strings/vocabularies
"""
from enum import Enum
from collections import Counter
import numpy as np
from . import console

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
    
    _UNK = "<unk>"

    def __init__(self, sentences, limit, use_tfidf=False):
        console.time("StringStore find vocab")
        sentences = [[word.lower() for word in sent] for sent in sentences]
        counts = Counter(word for sent in sentences for word in sent)
        if use_tfidf:
            n_contain = Counter()
            for sent in sentences:
                for w in set(sent):
                    n_contain[w] += 1
            def tfidf(word):
                return counts[word] * np.log(len(sentences) / (1 + n_contain[word]))
            self._id2w = sorted(counts.keys(), key=tfidf, reverse=True)[:limit-1]
        else:
            self._id2w = [word for word, _ in counts.most_common(limit-1)]
        console.time("StringStore find vocab")
        self._id2w.append(self._UNK)
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

    def __iter__(self):
        return iter(self._id2w)

    def word2id(self, word):
        """Return an integer ID for word. Word may be unk."""
        if word not in self._w2id:
            word = self._UNK
        return self._w2id[word]

    def id2word(self, i):
        """Return the word with ID=i. Requires 0 <= i < len(me)."""
        return self._id2w[i]

    def count_vector(self, word_iter):
        """Return a vector v with length equal to my vocabulary size,
        where v[i] = count of word with ID=i seen in word_iter.
        """
        return np.bincount([self.word2id(w) for w in word_iter], minlength=len(self))
