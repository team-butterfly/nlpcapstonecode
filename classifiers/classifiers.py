from itertools import chain

from utility import console, Emotion
from utility.strings import StringStore

import numpy as np
import tensorflow as tf

class Classifier():
    def train(self, inputs, true_labels, num_labels):
        """Trains the model parameters until they converge."""
        pass

    def predict(self, text):
        """Returns one of the Emotion constants."""
        pass


class LSTMClassifier(Classifier):
    """
    LSTMClassifier classifies sentences with some kind of LSTM.
    """
    def __init__(self):
        pass


class UnigramClassifier(Classifier):
    """
    UnigramClassifier classifies sentences by a simple bag-of-words model.
    """
    def __init__(self):
        pass

    def _encode_sentences(self, sentences):
        """Convert a sequence of sentences into a matrix of unigram counts."""
        return np.array([self._stringstore.count_vector(sent.split()) for sent in sentences])

    def train(self, inputs, true_labels, num_labels):
        """
        inputs: training inputs, a sequence of sentences
        true_labels: integer labels for each sentence
        num_labels: total number of labels for classification
        """
        # First, set up the vocabulary and such
        def word_iter():
            for sent in inputs:
                for word in sent.split():
                    yield word

        self._stringstore = StringStore(word_iter(), unk_threshold=2)
        self._vocab_size = len(self._stringstore)
        self._num_labels = num_labels

        console.log("UnigramClassifier.train: vocab:", self._stringstore.vocab())
        console.log("UnigramClassifier.train: vocabulary size:", self._vocab_size)
        console.log("UnigramClassifier.train: num labels:", self._num_labels)
        console.log("UnigramClassifier.train: raw inputs:\n", inputs)
        console.log("UnigramClassifier.train: encoded inputs:\n", self._encode_sentences(inputs))
        console.log("UnigramClassifier.train: true labels:\n", true_labels)

        # Build the Tensorflow computation graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Model inputs
            self._inputs = tf.placeholder(tf.float32, [None, self._vocab_size], name="word_counts")
            self._true_labels = tf.placeholder(tf.int32, [None], name="labels")
            
            # Model parameters
            self._w = tf.Variable(tf.truncated_normal([self._vocab_size, self._num_labels]),
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
            # Calculate average over every interval of this many steps
            avg_steps = 10
            # previous avg. loss, current avg. loss
            t = 0
            avg_loss_0 = avg_loss_1 = 0

            # We've converged when avg_loss_0 ~= avg_loss_1
            converged = False
            
            while not converged:
                t += 1
                x_batch, y_batch = self._encode_sentences(inputs), true_labels
                train_feed = {
                    self._inputs: x_batch,
                    self._true_labels: y_batch
                }
                [_, cur_loss] = sess.run([self._train_op, self._loss], train_feed)
                avg_loss_1 += cur_loss
                if t > 0 and t % avg_steps == 0:
                    avg_loss_1 /= avg_steps
                    avg_loss_0, avg_loss_1 = avg_loss_1, 0
                    converged = np.isclose(avg_loss_0, avg_loss_1, rtol=1e-02, atol=1e-03)

            # Temporary kludge until I figure out the right way to do this...
            self._w_saved = self._w.eval()
            self._b_saved = self._b.eval()

            console.log("UnigramClassifier.train: converged in {} steps.".format(t))
            console.log("UnigramClassifier.train: weights\n", self._w_saved)
            console.log("UnigramClassifier.train: biases\n", self._b_saved)

    def predict(self, text):
        with tf.Session(graph=self._graph) as sess:
            sess.run(self._init_op)
            feed = {
                self._inputs: self._encode_sentences(text),
                self._w: self._w_saved,
                self._b: self._b_saved
            }
            predictions = sess.run(self._predictions, feed_dict=feed)
            return predictions