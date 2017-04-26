from utility import console
from utility.strings import StringStore

from .classifier import Classifier
from .batch import Batch

import numpy as np

# Disable tensorflow warning/debugging log messages.
from os import environ; environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

class UnigramClassifier(Classifier):
    """
    ``UnigramClassifier` classifies sentences by a simple bag-of-words model.
    """
    def __init__(self, num_labels, unk_threshold=7):
        """
        Parameters:
            `unk_threshold` if a token appears less than this many times,
                it is not added to the classifer's vocabulary
        """
        self._num_labels = num_labels
        self._unk_threshold = unk_threshold

    def _encode_sentences(self, sentences):
        """Convert a list of token sequences into a matrix of unigram counts."""
        return np.array([self._stringstore.count_vector(sent) for sent in sentences])

    def train(self, sentences, true_labels, max_epochs=1000):
        """
        Parameters:
            `sentences` training inputs -- a list of lists of tokens
            `true_labels` list integer labels, one for each sentence
            `num_labels` total number of labels for classification
            `max_epochs` (default 1000) maximum number of training epochs to run
        """
        if max_epochs is None:
            raise ValueError("UnigramClassifier max_epochs cannot be None")

        # First, set up the vocabulary and such
        word_iter = (token for sent in sentences for token in sent)
        self._stringstore = StringStore(word_iter, unk_threshold=self._unk_threshold)
        self._vocab_size = len(self._stringstore)

        console.log("UnigramClassifier.train: vocabulary size is", self._vocab_size)

        # Set up minibatching
        encoded_inputs = self._encode_sentences(sentences)
        array_labels = np.array(true_labels)
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
            t = 0
            for t in range(max_epochs):
                batch = get_minibatch(50)
                train_feed = {
                    self._inputs:      batch.xs,
                    self._true_labels: batch.ys
                }
                sess.run([self._train_op], train_feed)

            # Temporary kludge until I figure out the right way to do this...
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
            predictions, probs = sess.run([self._predictions, self._softmax], feed_dict=feed)
            return predictions
