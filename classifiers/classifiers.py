from collections import namedtuple

from utility import console, Emotion
from utility.strings import StringStore

import numpy as np

# Disable tensorflow warning/debugging log messages.
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

Batch = namedtuple("Batch", ["xs", "ys", "lengths"])

class Classifier():
    def __init__(self):
        raise Exception("Not implemented")

    def train(self, inputs, true_labels):
        """`train` should train the model parameters until they converge."""
        raise Exception("Not implemented")

    def predict(self, inputs):
        """`predict` should return a list of Emotion constants."""
        raise Exception("Not implemented")


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
        # First, set up the vocabulary and such
        word_iter = (token for sent in sentences for token in sent)
        self._stringstore = StringStore(word_iter, unk_threshold=self._unk_threshold)
        self._vocab_size = len(self._stringstore)

        console.log("UnigramClassifier.train: vocabulary size is", self._vocab_size)

        # Set up minibatching
        encoded_inputs = self._encode_sentences(sentences)
        array_labels = np.array(true_labels)
        batch_idx = 0
        def get_batch(size):
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
            for t in range(max_epochs):
                batch = get_batch(50)
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


class LstmClassifier(Classifier):
    """
    LstmClassifier classifies sentences with a char-level LSTM.
    """
    def __init__(self, num_labels):
        self._num_labels  = num_labels
        self._hidden_size = 128 # Size of LSTM hidden state
        self._num_layers  = 1   # Number of LSTM layers
        self._vocab_size  = 256 # TODO: Only use the char set seen in training data?
        self._seq_len     = 160 # Max sequence length supported by LSTM

    def _encode_raw_inputs(self, raw_inputs):
        """Convert list of strings to list of arrays of ASCII values"""
        max_len = max(len(st) for st in raw_inputs)
        if max_len > self._seq_len:
            raise ValueError("Inputs can't exceed the LSTM's sequence length, which is {}".format(self._seq_len))

        return [np.array([ord(char) for char in sentence]) for sentence in raw_inputs]

    def train(self, raw_inputs, true_labels, max_epochs=1000):
        """
        Parameters:
            `raw_inputs` a list of raw tweets. i.e, a list of strings.
            `true_labels` a list of integer labels corresponding to each raw input.
            `max_epochs` (optional) max number of training batches to run.
        """
        char_inputs = self._encode_raw_inputs(raw_inputs)
        
        def get_batch(size, batch_idx=0):
            """
            Return a Batch, which is a named tuple of
                `x` an array with dims [size, LEN] where LEN<=seq_len,
                    each row is a sequence that will be input to the model,
                    consisting of an array of ASCII characters padded with zeros to reach LEN.
                `y` the correct output labels.
                `lengths` the true lengths of each input sequence (so that the LSTM can ignore the padding)
            """
            indices = (batch_idx + np.arange(size)) % len(char_inputs)
            max_length = max(len(char_inputs[i]) for i in indices)

            x = np.zeros([size, max_length], dtype=np.int32)
            y = np.zeros([size], dtype=np.int)
            lengths = np.zeros([size], dtype=np.int)
            
            for i in indices:
                pad = max_length - len(char_inputs[i])
                x[i] = np.append(char_inputs[i], np.zeros(pad, np.int))
                lengths[i] = len(char_inputs[i])
                y[i] = true_labels[i]

            batch_idx = (batch_idx + size) % len(char_inputs)
            return Batch(xs=x, ys=y, lengths=lengths)

        def dense(size_in, size_out):
            w_init = tf.truncated_normal([size_in, size_out], stddev=0.3)
            w = tf.Variable(w_init)
            b = tf.Variable(tf.zeros([size_out]))
            return lambda x: tf.nn.xw_plus_b(x, w, b)

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._batch_size   = tf.placeholder(tf.int32, name="batch_size")
            self._inputs       = tf.placeholder(tf.int32, [None, None], name="inputs")
            self._true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")
            self._labels       = tf.placeholder(tf.int32, [None], name="labels")

            self._one_hot_inputs = tf.one_hot(self._inputs, depth=self._vocab_size)

            self._cell = tf.contrib.rnn.LSTMCell(self._hidden_size, state_is_tuple=True)
            
            # Add multi-layeredness
            if self._num_layers > 1:
                self._cell = tf.contrib.rnn.MultiRNNCell(
                    [self._cell] * self._num_layers,
                    state_is_tuple=True)

            self._initial_state = self._cell.zero_state(self._batch_size, tf.float32)
            self._outputs, final_state_tuple = tf.nn.dynamic_rnn(
                self._cell,
                self._one_hot_inputs,
                initial_state=self._initial_state,
                sequence_length=self._true_lengths,
                dtype=tf.float32)

            self._final_states = self._outputs[:, -1, :]
            unembed = dense(self._hidden_size, self._num_labels)
            self._logits = unembed(self._final_states)

            self._loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels,
                logits=self._logits))


            self._softmax = tf.nn.softmax(self._logits)

            # Must cast tf.argmax to int32 because it returns int64
            self._labels_predicted = tf.cast(tf.argmax(self._softmax, axis=1), tf.int32)

            self._correct  = tf.equal(self._labels, self._labels_predicted)
            self._accuracy = tf.reduce_mean(tf.cast(self._correct, tf.float32))

            self._train_op = tf.train.AdamOptimizer().minimize(self._loss)
            self._init_op = tf.global_variables_initializer()

            # Save the model at the end of training -- only keep 1 checkpoint file at most.
            self._saver = tf.train.Saver(max_to_keep=1)

        # Begin training loop
        with tf.Session(graph=self._graph) as sess:
            sess.run(self._init_op)
            avg_loss = 0
            avg_steps = 10
            batch_size = 20
            t = 0
            while t <= max_epochs:
                print(".", end="", flush=True)
                batch = get_batch(batch_size)
                train_feed = {
                    self._batch_size:   batch_size,
                    self._inputs:       batch.xs,
                    self._true_lengths: batch.lengths,
                    self._labels:       batch.ys
                }
                [_, cur_loss] = sess.run([self._train_op, self._loss], train_feed)
                avg_loss += cur_loss
                t += 1

                # Every few steps, report average loss
                if t % avg_steps == 0:
                    avg_loss /= (avg_steps if t > 0 else 1)
                    print("Step", t)
                    print("\tAverage loss:", avg_loss)
            self._saved_path = self._saver.save(sess, "ckpts/lstm", global_step=t)
            console.log("Lstm saved to", self._saved_path)

    def predict(self, raw_inputs):

        n_inputs = len(raw_inputs)
        char_inputs = self._encode_raw_inputs(raw_inputs)
        max_length = max(len(seq) for seq in char_inputs)

        x = np.zeros([n_inputs, max_length], dtype=np.int32) # TODO use int32 above
        lengths = np.zeros(n_inputs, dtype=np.int32)
        
        for i, char_seq in enumerate(char_inputs):
            pad = max_length - len(char_seq)
            x[i] = np.pad(char_seq, (0, pad), "constant")
            lengths[i] = len(char_seq)

        with tf.Session(graph=self._graph) as sess:
            #sess.run(self._init_op)
            self._saver.restore(sess, self._saved_path)
            feed = {
                self._inputs: x,
                self._true_lengths: lengths,
                self._batch_size: n_inputs
            }
            [predictions] = sess.run([self._labels_predicted], feed)
        return predictions