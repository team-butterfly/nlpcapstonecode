import glob
import html
import json
import nltk
import os
import random

from utility import console, Emotion

from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.moses import MosesTokenizer
from nltk.tokenize.repp import ReppTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

# e.g. ds = TweetsDataSource(file_glob="data/tweets.v3.*.txt", random_seed=5)
class TweetsDataSource(object):

    _tokenizer = TweetTokenizer()

    def _clean_text(text, escape_unicode=False):
        if escape_unicode:
            # Un-escape unicode characters
            text = bytes(text, 'ascii').decode('unicode_escape')

        # Replace newline characters with two spaces
        text = text.replace('\n', '  ')

        # Un-escape HTML entities
        text = html.unescape(text)

        return text

    def tokenize(sent):
        return TweetsDataSource._tokenizer.tokenize(sent)

    def __init__(self, *args, **kwargs):
        pct_test = 0.10
        if 'pct_test' in kwargs:
            pct_test = kwargs['pct_test']

        random_seed = None
        if 'random_seed' in kwargs:
            random_seed = kwargs['random_seed']

        filenames = []
        filenames += args
        if 'file_glob' in kwargs:
            filenames += glob.glob(kwargs['file_glob'])

        tokenize = TweetTokenizer().tokenize
        if 'tokenizer' in kwargs:
            if kwargs['tokenizer'] == 'wordpunct':
                tokenize = wordpunct_tokenize
            elif kwargs['tokenizer'] == 'word':
                tokenize = word_tokenize
            elif kwargs['tokenizer'] == 'treebank':
                tokenize = TreebankWordTokenizer().tokenize
            elif kwargs['tokenizer'] == 'moses':
                tokenize = MosesTokenizer().tokenize
            elif kwargs['tokenizer'] == 'tweet':
                tokenize = TweetTokenizer().tokenize

        self._raw_inputs = []
        self._inputs = []
        emotions = []
        for filename in filenames:
            if not os.path.isfile(filename):
                console.warn("Not a file: " + filename)
                continue

            # console.info("Parsing " + filename)

            with open(filename, 'r') as f:
                lines = f.readlines()

            new_inputs = []
            new_emotions = []
            if lines[0].rstrip() == 'v.5/16-tok':
                lines = lines[1:]
                tweets = [json.loads(line.rstrip()) for line in lines]
                new_inputs = [TweetsDataSource._clean_text(tweet['text'].strip()) for tweet in tweets]
                new_emotions = [Emotion[tweet['tag']] for tweet in tweets]
                self._raw_inputs += new_inputs
                self._inputs += [text.split() for text in new_inputs]
            elif lines[0].rstrip() == 'v.4/21':
                lines = lines[1:]
                tweets = [json.loads(line.rstrip()) for line in lines]
                new_inputs = [TweetsDataSource._clean_text(tweet['text'].strip()) for tweet in tweets]
                new_emotions = [Emotion[tweet['tag']] for tweet in tweets]
                self._raw_inputs += new_inputs
                self._inputs += [tokenize(text) for text in new_inputs]
            else:
                new_inputs = [
                    TweetsDataSource._clean_text(lines[i + 3].rstrip(), True)
                    for i in range(0, len(lines), 5)]
                new_emotions = [Emotion[lines[i + 2].rstrip()] for i in range(0, len(lines), 5)]
                self._raw_inputs += new_inputs
                self._inputs += [tokenize(text) for text in new_inputs]

            emotions += new_emotions

        num_inputs = len(self._inputs)

        self._index_emotion = list(set(emotions))
        self._emotion_index = {l: l.value for l in self._index_emotion}
        self._num_labels = len(self._index_emotion)
        self._labels = [self._emotion_index[emotion] for emotion in emotions]

        num_test = int(round(len(self._inputs) * pct_test))

        random.seed(random_seed)
        self._test_indexes = sorted(random.sample(range(num_inputs), num_test))

        console.info("Initialized data source with " + str(num_inputs) + " tweets")

    def num_tokens(self):
        return sum([len(i) for i in self._inputs])

    def vocab_size(self):
        from collections import Counter
        c = Counter()
        for i in self._inputs:
            c.update(i)
        return len(c)

    def vocab(self):
        s = set()
        for i in self._inputs:
            s.update(i)
        return s

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def train_inputs(self):
        return [s for i, s in enumerate(self._inputs) if i not in self._test_indexes]

    @property
    def train_raw_inputs(self):
        return [s for i, s in enumerate(self._raw_inputs) if i not in self._test_indexes]

    @property
    def test_inputs(self):
        return [self._inputs[i] for i in self._test_indexes]

    @property
    def test_raw_inputs(self):
        return [self._raw_inputs[i] for i in self._test_indexes]

    @property
    def train_labels(self):
        return [s for i, s in enumerate(self._labels) if i not in self._test_indexes]

    @property
    def test_labels(self):
        return [self._labels[i] for i in self._test_indexes]

    def decode_labels(self, labels):
        return [self._index_emotion[i].name for i in labels]
