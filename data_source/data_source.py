
import nltk


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
