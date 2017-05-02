from collections import defaultdict, deque
from utility import console, Emotion
from .classifier import Classifier
from .batch import Batch, Minibatcher
import numpy as np
import tensorflow as tf


class _EmoLexGraph():

    def __init__(self, input_dim, output_dim):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, input_dim])
            self.labels = tf.placeholder(tf.int32, [None]) # true labels
            
            self.w = tf.Variable(
                    tf.truncated_normal([input_dim, output_dim], stddev=2/np.sqrt(input_dim)))
            self.b = tf.Variable(tf.zeros([output_dim]))

            self.h0 = tf.nn.elu(tf.nn.xw_plus_b(self.inputs, self.w, self.b))
            self.w0 = tf.Variable(
                    tf.truncated_normal([output_dim, output_dim], stddev=2/np.sqrt(output_dim)))
            self.b0 = tf.Variable(tf.zeros([output_dim]))

            self.h1 = tf.nn.elu(tf.nn.xw_plus_b(self.h0, self.w0, self.b0))
            self.w1 = tf.Variable(
                    tf.truncated_normal([output_dim, output_dim], stddev=2/np.sqrt(output_dim)))
            self.b1 = tf.Variable(tf.zeros([output_dim]))

            self.logits = tf.nn.xw_plus_b(self.h1, self.w1, self.b1)
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

    def __init__(self, emolex_path):
        self._index_emotion = self._keys
        self._emotion_index = {key: i for i, key in enumerate(self._keys)}
        self._map = defaultdict(lambda: np.zeros(len(self._keys), np.int32))
        self._g = _EmoLexGraph(len(self._keys), len(Emotion))

        with open(emolex_path) as f:
            f.readline()
            for line in f.readlines():
                word, emotion, assoc = line.split("\t")
                self._map[word][self._emotion_index[emotion]] = assoc


    def _to_vector(self, sent):
        counts = np.zeros(len(self._keys), np.int32)
        for word1 in sent:
            word = word1.lower()
            if word in self._map:
                counts += self._map[word]
        return counts # TODO look at not-unit vector


    def _encode_inputs(self, input_tokens):
        input_counts = np.empty([len(input_tokens), len(self._keys)], np.int32)
        bad = 0
        for i, tokens in enumerate(input_tokens):
            input_counts[i] = self._to_vector(tokens)
            if input_counts[i].sum() == 0:
                bad += 1
        console.warn("{} of {} inputs are completely zero.".format(bad, len(input_tokens)))
        return input_counts


    def train(self, input_tokens, labels):
        train_data = Batch(xs=self._encode_inputs(input_tokens), ys=np.array(labels), lengths=None)
        minibatcher = Minibatcher(train_data)

        losses = deque(maxlen=50)

        with tf.Session(graph=self._g.graph) as sess:
            sess.run(self._g.init_op)
            idxs = np.arange(64)
            for t in range(1, 10001):
                batch = minibatcher.next(64)
                np.random.shuffle(idxs)
                batch = Batch(batch.xs[idxs], batch.ys[idxs], None)
                train_feed = {
                    self._g.inputs: batch.xs,
                    self._g.labels: batch.ys
                }
                _, loss = sess.run((self._g.train_op, self._g.loss), train_feed)
                losses.append(loss)
                if t % 50 == 0:
                    console.log("{:.6f}".format(np.mean(losses)))
            self._w_save = sess.run(self._g.w)
            self._b_save = sess.run(self._g.b)
            self._w0_save = sess.run(self._g.w0)
            self._b0_save = sess.run(self._g.b0)
            self._w1_save = sess.run(self._g.w1)
            self._b1_save = sess.run(self._g.b1)


    def predict(self, input_tokens):
        return np.argmax(self.predict_soft(input_tokens), axis=1)


    def predict_soft(self, input_tokens):
        feed = {
            self._g.w: self._w_save,
            self._g.b: self._b_save,
            self._g.w0: self._w0_save,
            self._g.b0: self._b0_save,
            self._g.w1: self._w1_save,
            self._g.b1: self._b1_save,
            self._g.inputs: self._encode_inputs(input_tokens)
        }
        with tf.Session(graph=self._g.graph) as sess:
            return sess.run(self._g.probabilities, feed)
