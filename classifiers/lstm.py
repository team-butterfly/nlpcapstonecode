"""
Defines a TensorFlow computation graph for a character-level LSTM
"""
from datetime import datetime
from collections import deque
import numpy as np
import tensorflow as tf

from utility import console
from .classifier import Classifier
from .batch import Minibatcher
from . import lstm_util



class _LstmGraph():

    def __init__(self):
        console.info("Building LSTM graph")
        self.root = tf.Graph()
        with self.root.as_default():
            # Model inputs
            self.batch_size   = tf.placeholder(tf.int32, name="batch_size")
            self.inputs       = tf.placeholder(tf.int32, [None, None], name="inputs")
            self.labels       = tf.placeholder(tf.int32, [None], name="labels")
            self.true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")

            self.one_hot_inputs = tf.one_hot(self.inputs, depth=lstm_util.VOCAB_SIZE)

            # If true, dropout is applied to LSTM cell inputs and outputs.
            self.use_dropout = tf.placeholder(tf.bool, name="use_dropout")
            self.keep_prob = tf.constant(0.5, name="keep_prob")

            # if self.use_dropout, then _keep_prob else 1.0
            self.keep_prob_conditional = tf.cond(
                self.use_dropout, 
                lambda: self.keep_prob,
                lambda: tf.constant(1.0))

            # The LSTM cell and RNN itself
            self.cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(lstm_util.HIDDEN_SIZE, state_is_tuple=True),
                input_keep_prob=self.keep_prob_conditional,
                output_keep_prob=self.keep_prob_conditional)

            # Add multi-layeredness
            if lstm_util.NUM_LAYERS > 1:
                self.cell = tf.contrib.rnn.MultiRNNCell([self.cell] * lstm_util.NUM_LAYERS, state_is_tuple=True)

            self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
            self.outputs, self.final_state_tuple = tf.nn.dynamic_rnn(
                self.cell,
                self.one_hot_inputs,
                initial_state=self.initial_state,
                sequence_length=self.true_lengths,
                dtype=tf.float32)

            self.final_states = self.final_state_tuple.h

            # Weights and biases for the fully-connected layer that
            # projects the final LSTM state down to the size of the label space
            self.w = tf.Variable(
                tf.truncated_normal(
                    [lstm_util.HIDDEN_SIZE, lstm_util.NUM_LABELS],
                    stddev=2/np.sqrt(lstm_util.HIDDEN_SIZE)),
                name="dense_weights")
            self.b = tf.Variable(tf.zeros(lstm_util.NUM_LABELS), name="dense_biases")

            self.logits = tf.nn.xw_plus_b(self.final_states, self.w, self.b)
            self.softmax = tf.nn.softmax(self.logits)

            # Must cast tf.argmax to int32 because it returns int64
            self.labels_predicted = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)

            # Loss function is mean cross-entropy
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=self.logits))

            # When executed, train op runs one step of gradient descent.
            # Global step counter (incremented by the AdamOptimizer on each gradient update)
            self.step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.step)
            # Init op initalizes variables to their default values
            self.init_op = tf.global_variables_initializer()

            # The Saver handles saving all model parameters at checkpoints and
            # restoring them later
            self.vars_saved = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.saver = tf.train.Saver(self.vars_saved, max_to_keep=10)


class LstmClassifier(Classifier):
    
    def __init__(self):
        self._g = _LstmGraph()

    def _restore(self, session):
        latest_ckpt = tf.train.latest_checkpoint("./ckpts")
        if latest_ckpt is None:
            raise ValueError("Latest checkpoint does not exist")

        self._g.saver.restore(session, latest_ckpt)
        console.info("Restored model from", latest_ckpt)

    def train(self,
        raw_inputs,
        true_labels,
        num_epochs=None,
        continue_previous=True,
        save_every_n_epochs=5,
        save_hook=lambda epoch, step: None):
        """
        Args:
            `raw_inputs` a list of raw tweets. i.e, a list of strings.
            `true_labels` a list of integer labels corresponding to each raw input.
            `num_epochs` number of training epochs to run. If None, train forever.
            `continue_previous` if `True`, load params from latest checkpoint and
                continue training from there.
            `save_every_n_epochs`
            `save_hook` if given, a callable that will be called with parameters
                (epoch number, step number) whenever a new checkpoint is saved
        """
        char_inputs = lstm_util.encode_raw_inputs(raw_inputs)
        train_data = lstm_util.make_batch(char_inputs, true_labels)

        train_feed = {
            self._g.use_dropout:  True,
            self._g.batch_size:   None,
            self._g.inputs:       None,
            self._g.labels:       None,
            self._g.true_lengths: None
        }

        epochs_done = 0
        losses = deque(maxlen=10)
        minibatcher = Minibatcher(train_data)

        with tf.Session(graph=self._g.root) as sess:
            try:
                self._restore(sess)
            except Exception as e:
                console.warn("Failed to restore from previous checkpoint:", e)
                console.warn("The model will be initialized from scratch.")
                sess.run(self._g.init_op)

            while True:
                if num_epochs is not None and minibatcher.cur_epoch > num_epochs:
                    break

                batch = minibatcher.next(128)
                train_feed[self._g.batch_size]   = len(batch.xs)
                train_feed[self._g.inputs]       = batch.xs
                train_feed[self._g.labels]       = batch.ys
                train_feed[self._g.true_lengths] = batch.lengths

                [_, cur_loss, step] = sess.run([self._g.train_op, self._g.loss, self._g.step], train_feed)
                losses.append(cur_loss)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.info("\tAverage loss (last {} steps): {:.4f}".format(len(losses), np.mean(losses)))
                    epochs_done += 1
                    if epochs_done % save_every_n_epochs == 0:
                        saved_path = self._g.saver.save(sess, "./ckpts/lstm", global_step=self._g.step)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tCheckpoint saved to {}".format(datetime.now(), saved_path)
                            + console.colors.END)
                        save_hook(epochs_done, step)
                else:
                    # Print some stuff so we know it's making progress
                    label = "Global Step {}".format(step)
                    console.progress_bar(label, minibatcher.epoch_progress, 60)


    def predict(self, raw_inputs):
        return np.argmax(self.predict_soft(raw_inputs), axis=1)


    def predict_soft(self, raw_inputs):
        with tf.Session(graph=self._g.root) as sess:
            self._restore(sess)
            input_as_chars = lstm_util.encode_raw_inputs(raw_inputs)
            batch = lstm_util.make_batch(input_as_chars, labels=None)
            feed = {
                self._g.use_dropout:  False,
                self._g.batch_size:   len(batch.xs),
                self._g.inputs:       batch.xs,
                self._g.true_lengths: batch.lengths
            }
            soft_labels = sess.run(self._g.softmax, feed)
        return soft_labels
