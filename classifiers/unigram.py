from utility import console, Emotion
from utility.strings import StringStore

from .classifier import Classifier
from .utility import Batch

from data_source.tokenize import tokenize_tweet

import numpy as np

import tensorflow as tf

class UnigramClassifier(Classifier):
    """
    ``UnigramClassifier` classifies sentences by a simple bag-of-words model.
    """
    def __init__(self, vocab_size=50000):
        """
        Args:
            `vocab_size` a hard limit on vocabulary size
        """
        self._num_labels = len(Emotion)
        self._vocab_size = vocab_size

    def _encode_sentences(self, sentences):
        """Convert a list of token sequences into a matrix of unigram counts."""
        out = np.empty([len(sentences), len(self._stringstore)], dtype=np.float32)
        for i, sent in enumerate(sentences):
            out[i] = self._stringstore.count_vector(tokenize_tweet(sent))
        return out

    def train(self, data_source, max_epochs=1000):
        """
        Args:
            `max_epochs` (default 1000) maximum number of training epochs to run
        """
        if max_epochs is None:
            raise ValueError("UnigramClassifier max_epochs cannot be None")

        # First, set up the vocabulary and such
        sentences = data_source.train_raw_inputs
        self._stringstore = StringStore(map(tokenize_tweet, sentences), self._vocab_size)

        console.log("UnigramClassifier.train: vocabulary size is", len(self._stringstore))

        # Set up minibatching
        encoded_inputs = self._encode_sentences(sentences)
        array_labels = data_source.train_labels
        batch_idx = 0
        def get_minibatch(size):
            nonlocal batch_idx
            indices = (np.arange(size) + batch_idx) % len(encoded_inputs)
            batch_idx = (batch_idx + size) % len(encoded_inputs)
            return Batch(xs=encoded_inputs[indices], ys=array_labels[indices], lengths=None)

        # Build the Tensorflow computation graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Model inputs
            with tf.device("/cpu:0"):
                self._inputs = tf.placeholder(tf.float32, [None, self._vocab_size], name="word_counts")
                self._true_labels = tf.placeholder(tf.int32, [None], name="labels")

                # Model parameters
                self._w = tf.Variable(tf.zeros([self._vocab_size, self._num_labels]),
                                      dtype=tf.float32,
                                      name="weights")
                self._b = tf.Variable(tf.zeros([self._num_labels]),
                                      dtype=tf.float32,
                                      name="bias")

                self._logits = tf.nn.xw_plus_b(self._inputs, self._w, self._b)
                self._softmax = tf.nn.softmax(self._logits)
                self._predictions = tf.argmax(self._softmax, axis=1)

                self._loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self._true_labels,
                        logits=self._logits))

                self._train_op = tf.train.AdamOptimizer().minimize(self._loss)
                self._init_op = tf.global_variables_initializer()

        # Begin the training loop
        with tf.Session(graph=self._graph) as sess:
            sess.run(self._init_op)
            console.info("Begin train loop")
            t = 0
            for t in range(max_epochs):
                batch = get_minibatch(50)
                train_feed = {
                    self._inputs:      batch.xs,
                    self._true_labels: batch.ys
                }
                sess.run(self._train_op, train_feed)

            self._w_saved = self._w.eval()
            self._b_saved = self._b.eval()

    def predict(self, sentences):
        with tf.Session(graph=self._graph) as sess:
            sess.run(self._init_op)
            feed = {
                self._inputs: self._encode_sentences(sentences),
                self._w: self._w_saved,
                self._b: self._b_saved
            }
            predictions = sess.run(self._predictions, feed_dict=feed)
            return predictions
