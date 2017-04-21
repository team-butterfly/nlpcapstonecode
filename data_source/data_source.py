
import html
import json
import nltk
import random

from nltk.tokenize import TweetTokenizer
from utility import Emotion


class FakeDataSource(object):

    def __init__(self):
        self._num_labels = 4
        self._train_inputs = [nltk.word_tokenize(raw_text) for raw_text in (
            "I love eggs for breakfast",
            "Flies love fruit",
            "I love bacon",
            "I love coffee",
            "I love chocolate",
            "I love, love chocolate and coffee"
            "I like spam sandwiches",
            "I like peas and onions",
            "I like water",
            "I like ice cream, especially chocolate",
            "I hate strawberry milk",
            "I hate swiss cheese",
            "I hate vegetables, especially asparagus",
            "I hate fruit",
            "I am Python",
            "I have no brain",
            "Chickens lay eggs",
            "Sam eats spam",
            "I drink water all day",
            "I drink coffee in the morning"
        )]

        self._test_inputs = [nltk.word_tokenize(raw_text) for raw_text in (
            "Anyone with a brain knows that dogs like coffee",
            "I ate spam for dinner",
            "We all love NLP",
            "Cats really love milk",
            "Today is Wednesday",
            "Everyone knows that snakes love eggs",
            "Cats absolutely hate the sensation of water on their skin",
        )]

    @property
    def num_labels(self):
        return self._num_labels

    def _generate_labels(self, sentences):
        return [
            3 if "love" in sentence else
            2 if "like" in sentence else
            1 if "hate" in sentence else
            0
            for sentence in sentences
        ]

    @property
    def train_inputs(self):
        return self._train_inputs

    @property
    def test_inputs(self):
        return self._test_inputs

    @property
    def train_labels(self):
        """Get the labels of training data"""
        return self._generate_labels(self._train_inputs)

    @property
    def test_labels(self):
        """Get the labels of test data"""
        return self._generate_labels(self._test_inputs)

    def decode_labels(self, labels, meanings=("neutral", "hatred", "enjoyment", "endearment")):
        return [meanings[i] for i in labels]


class TweetsDataSource(object):

    _tokenizer = TweetTokenizer()

    def _parse_tok(line, escape_unicode=False):
        if escape_unicode:
            # Un-escape unicode characters
            line = bytes(line, 'ascii').decode('unicode_escape')

        # Replace newline characters with two spaces
        line = line.replace('\n', '  ')

        # Un-escape HTML entities
        line = html.unescape(line)

        return TweetsDataSource._tokenizer.tokenize(line)

    def __init__(self, filename, pct_test=0.10, random_seed=None):
        with open(filename, 'r') as f:
            lines = f.readlines()

        if lines[0].rstrip() == 'v.4/21':
            lines = lines[1:]
            tweets = [json.loads(line.rstrip()) for line in lines]
            self._inputs = [TweetsDataSource._parse_tok(tweet['text']) for tweet in tweets]
            emotions = [Emotion[tweet['tag']] for tweet in tweets]
        else:
            self._inputs = [TweetsDataSource._parse_tok(lines[i + 3].rstrip(), True) for i in range(0, len(lines), 5)]
            emotions = [Emotion[lines[i + 2].rstrip()] for i in range(0, len(lines), 5)]

        num_inputs = len(self._inputs)

        self._index_emotion = list(set(emotions))
        self._emotion_index = {l: i for i, l in enumerate(self._index_emotion)}
        self._num_labels = len(self._index_emotion)
        self._labels = [self._emotion_index[emotion] for emotion in emotions]

        num_test = int(round(len(self._inputs) * pct_test))

        random.seed(random_seed)
        self._test_indexes = sorted(random.sample(range(num_inputs), num_test))

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def train_inputs(self):
        return [s for i, s in enumerate(self._inputs) if i not in self._test_indexes]

    @property
    def test_inputs(self):
        return [self._inputs[i] for i in self._test_indexes]

    @property
    def train_labels(self):
        return [s for i, s in enumerate(self._labels) if i not in self._test_indexes]

    @property
    def test_labels(self):
        return [self._labels[i] for i in self._test_indexes]

    def decode_labels(self, labels):
        return [self._index_emotion[i].name for i in labels]
