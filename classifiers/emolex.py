from collections import defaultdict, deque
from utility import console, Emotion
from .classifier import Classifier
from .utility import Batch, Minibatcher
import numpy as np
import tensorflow as tf


class _EmoLexGraph():

    def __init__(self, input_dim, output_dim):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                self.inputs = tf.placeholder(tf.float32, [None, input_dim])
                self.labels = tf.placeholder(tf.int32, [None]) # true labels
                
                self.w = tf.Variable(
                        tf.random_uniform([input_dim, output_dim],
                            minval=-6/np.sqrt(input_dim + output_dim),
                            maxval=-6/np.sqrt(input_dim + output_dim)))
                self.b = tf.Variable(tf.zeros([output_dim]))

                self.logits = tf.nn.xw_plus_b(self.inputs, self.w, self.b)
                self.probabilities = tf.nn.softmax(self.logits)

                self.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

                self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
                self.init_op = tf.global_variables_initializer()


class EmoLexBowClassifier(Classifier):
    """EmoLexBowClassifier implements a bag-of-words approach on the emotional lexicon."""
    _keys = (
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust"
    )

    def __init__(self):
        self._index_emotion = self._keys
        self._emotion_index = {key: i for i, key in enumerate(self._keys)}
        self._map = defaultdict(lambda: np.zeros(len(self._keys), np.int32))
        self._g = _EmoLexGraph(len(self._keys), len(Emotion))

        with open("emolex/emolex.txt", "r") as f:
            f.readline()
            for line in f.readlines():
                word, emotion, assoc = line.split("\t")
                self._map[word][self._emotion_index[emotion]] = int(assoc)


    def _to_vector(self, sent):
        counts = np.zeros(len(self._keys), np.int32)
        for word1 in sent:
            word = word1.lower()
            if word in self._map:
                counts += self._map[word]
        if (counts > 0).all():
            return counts / np.linalg.norm(counts)
        else:
            return counts


    def _encode_inputs(self, input_tokens):
        input_counts = np.empty([len(input_tokens), len(self._keys)], np.int32)
        bad = 0
        for i, tokens in enumerate(input_tokens):
            input_counts[i] = self._to_vector(tokens)
            if all(tok not in self._map for tok in tokens):
                bad += 1
        if bad > 0:
            console.warn("{} of {} inputs are completely OOV".format(bad, len(input_tokens)))
        return input_counts


    def train(self, data_source):
        input_tokens = data_source.train_inputs
        labels = data_source.train_labels
        train_data = Batch(xs=self._encode_inputs(input_tokens), ys=np.array(labels), lengths=None)
        minibatcher = Minibatcher(train_data)

        with tf.Session(graph=self._g.graph) as sess:
            sess.run(self._g.init_op)
            for t in range(1, 10001):
                batch = minibatcher.next(64)
                train_feed = {
                    self._g.inputs: batch.xs,
                    self._g.labels: batch.ys
                }
                _, loss = sess.run((self._g.train_op, self._g.loss), train_feed)
            self._w_save = sess.run(self._g.w)
            self._b_save = sess.run(self._g.b)


    def predict(self, input_tokens):
        return np.argmax(self.predict_soft(input_tokens), axis=1)


    def predict_soft(self, input_tokens):
        feed = {
            self._g.w: self._w_save,
            self._g.b: self._b_save,
            self._g.inputs: self._encode_inputs(input_tokens)
        }
        with tf.Session(graph=self._g.graph) as sess:
            return sess.run(self._g.probabilities, feed)
