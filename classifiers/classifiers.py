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

class UnigramClassifier(Classifier):
    """
    UnigramClassifier classifies sentences by a simple bag-of-words model.
    """
    def __init__(self, num_labels, unk_threshold=5):
        self._num_labels = num_labels
        self._unk_threshold = unk_threshold

    def _encode_sentences(self, sentences):
        """Convert a sequence of sentences into a matrix of unigram counts."""
        return np.array([self._stringstore.count_vector(sent) for sent in sentences])

    def train(self, inputs, true_labels, max_epochs=1000):
        """
        inputs: training inputs, a sequence of sentences
        true_labels: integer labels for each sentence
        num_labels: total number of labels for classification

        Mandatory keyword args:
        unk_threshold: if a token appears less than this many times, it is not added to the classifer's vocabulary
        max_epochs: maximum number of training epochs to run
        """
        # First, set up the vocabulary and such
        def word_iter():
            for sent in inputs:
                for token in sent:
                    yield token

        self._stringstore = StringStore(word_iter(), unk_threshold=self._unk_threshold)
        self._vocab_size = len(self._stringstore)

        console.log("UnigramClassifier.train: vocabulary size:", self._vocab_size)
        console.log("UnigramClassifier.train: num labels:", self._num_labels)

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

        # Batching
        encoded_inputs = self._encode_sentences(inputs)
        array_labels = np.array(true_labels)
        batch_idx = 0
        def get_batch(size):
            nonlocal batch_idx
            indices = (np.arange(size) + batch_idx) % len(inputs)
            batch_idx = (batch_idx + size) % len(inputs)
            return encoded_inputs[indices], array_labels[indices]

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
                x_batch, y_batch = get_batch(50)
                train_feed = {
                    self._inputs: x_batch,
                    self._true_labels: y_batch
                }
                [_, cur_loss] = sess.run([self._train_op, self._loss], train_feed)
                t += 1

                if t >= max_epochs:
                    break

                avg_loss_1 += cur_loss
                if t > 0 and t % avg_steps == 0:
                    avg_loss_1 /= avg_steps
                    converged = np.isclose(avg_loss_0, avg_loss_1, rtol=1e-04, atol=1e-05)
                    avg_loss_0, avg_loss_1 = avg_loss_1, 0

            # Temporary kludge until I figure out the right way to do this...
            self._w_saved = self._w.eval()
            self._b_saved = self._b.eval()

            console.log("UnigramClassifier.train: terminated training after {} steps.".format(t))

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

class LSTMClassifier(Classifier):
    """
    LSTMClassifier classifies sentences with some kind of LSTM.
    """



    def __init__(self, num_labels, hidden_size):
        self._num_labels = num_labels
        self._hidden_size = 128 # Size of LSTM hidden state
        self._num_layers = 1 # Number of LSTM layers
        self._vocab_size = 256 # TODO: Only use the char set seen in training data.

    def train(self, inputs, true_labels, max_epochs=1000):
        def embeddings(self, xs):
            """
            Input:  an iterable of integer lexeme ids
            Return: a matrix of word embeddings with dims [num_words, embed_size]
            """
            n = len(xs)
            output = np.zeros([n, EMBED_SIZE], np.bool)
            output[np.arange(n), xs] = True
            return output

        def yield_all_pairs(self):
            i = 0
            while True:
                # i = np.random.randint(NUM_SAMPLES)
                hist_indices = (i + np.arange(HIST_LEN)) % NUM_SAMPLES
                hist_indices = i + np.arange(HIST_LEN)
                batch_x = corpus[hist_indices]
                batch_y = corpus[i+HIST_LEN]
                i = (i+1) % NUM_SAMPLES
                yield embeddings(batch_x), batch_y

        sample_iter = yield_all_pairs()

        def get_batch(size):
            x = np.empty([size, HIST_LEN, EMBED_SIZE], dtype=np.float32)
            y = np.empty([size], dtype=np.int)
            for i in range(size):
                x[i], y[i] = next(sample_iter)
            return x, y

        def dense(size_in, size_out):
            w_init = tf.truncated_normal([size_in, size_out], stddev=0.3)
            w = tf.Variable(w_init)
            b = tf.Variable(tf.zeros([size_out]))
            return lambda x: tf.nn.xw_plus_b(x, w, b)

        graph = tf.Graph()
        with graph.as_default():
            batch_size = tf.placeholder(tf.int32)
            inputs     = tf.placeholder(tf.float32,
                                        [None, None, self._embed_size],
                                        name="inputs")
            true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")
            labels = tf.placeholder(tf.int32,
                                    [None], 
                                    name="labels")

            temperature = tf.placeholder(tf.float32, [1], "temperature")

            cell = tf.contrib.rnn.LSTMCell(self._hidden_size, state_is_tuple=True)
            
            # Add multi-layeredness
            if self._num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * self._num_layers, state_is_tuple=True)

            initial_state = cell.zero_state(batch_size, tf.float32)
            states, final_state_tuple = tf.nn.dynamic_rnn(
                cell,
                inputs,
                initial_state=initial_state,
                sequence_length=true_lengths,
                dtype=tf.float32)

            final_states = states[:, -1, :]
            unembed = dense(self._hidden_size, self._vocab_size)
            logits = unembed(final_states)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits))

            optimizer  = tf.train.AdamOptimizer()
            train_step = optimizer.minimize(loss)

            softmax = tf.nn.softmax(logits)

            # cast tf.argmax to int32 because it returns int64
            next_words_predicted = tf.cast(
                tf.argmax(softmax, axis=1),
                tf.int32)

            sample_outputs = tf.nn.softmax(logits / temperature)
            correct  = tf.equal(labels, next_words_predicted)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            avg_loss = 0
            for t in range(max_epochs + 1):
                print(".", end="")
                batch_size = 20
                batch_inputs, batch_labels = get_batch(batch_size)
                train_feed = {
                    batch_size: batch_size,
                    inputs: batch_inputs,
                    true_lengths: np.repeat(HIST_LEN, batch_size),
                    labels: batch_labels
                }
                _, cur_loss = sess.run([train_step, loss], train_feed)
                avg_loss += cur_loss

                if t % avg_steps != 0:
                    continue

                # Every few steps, report average loss

                avg_loss /= (AVG_STEPS if t > 0 else 1)
                print("Step", t)
                print("\tAverage loss:", avg_loss)

    def predict(self, sentences):
        return [0]