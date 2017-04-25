"""
Defines a TensorFlow computation graph for a character-level LSTM
"""
from collections import deque
import numpy as np
import tensorflow as tf

from utility import console
from .classifier import Classifier
from . import lstm_util

console.info("Building LSTM graph")

# Model inputs
_batch_size   = tf.placeholder(tf.int32, name="batch_size")
_inputs       = tf.placeholder(tf.int32, [None, None], name="inputs")
_labels       = tf.placeholder(tf.int32, [None], name="labels")
_true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")

_one_hot_inputs = tf.one_hot(_inputs, depth=lstm_util.VOCAB_SIZE)

# Global step counter (incremented by the AdamOptimizer on each gradient update)
_step = tf.Variable(0, name="global_step", trainable=False)

# The LSTM cell and RNN itself
_cell = tf.contrib.rnn.LSTMCell(lstm_util.HIDDEN_SIZE, state_is_tuple=True)

# Add multi-layeredness
if lstm_util.NUM_LAYERS > 1:
    _cell = tf.contrib.rnn.MultiRNNCell([_cell] * lstm_util.NUM_LAYERS, state_is_tuple=True)

_initial_state = _cell.zero_state(_batch_size, tf.float32)
_outputs, _final_state_tuple = tf.nn.dynamic_rnn(
    _cell,
    _one_hot_inputs,
    initial_state=_initial_state,
    sequence_length=_true_lengths,
    dtype=tf.float32)

_final_states = _final_state_tuple.h

# Weights and biases for the fully-connected layer that
# projects the final LSTM state down to the size of the label space
_w = tf.Variable(
    tf.truncated_normal(
        [lstm_util.HIDDEN_SIZE, lstm_util.NUM_LABELS],
        stddev=2/np.sqrt(lstm_util.HIDDEN_SIZE)),
    name="dense_weights")
_b = tf.Variable(tf.zeros(lstm_util.NUM_LABELS), name="dense_biases")

_logits = tf.nn.xw_plus_b(_final_states, _w, _b)
_softmax = tf.nn.softmax(_logits)

# Must cast tf.argmax to int32 because it returns int64
_labels_predicted = tf.cast(tf.argmax(_softmax, axis=1), tf.int32)

# Loss function is mean cross-entropy
_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=_labels,
    logits=_logits))

# These tensors calculate performance metrics
_correct  = tf.equal(_labels, _labels_predicted)
_accuracy = tf.reduce_mean(tf.cast(_correct, tf.float32))

# When executed, train op runs one step of gradient descent.
_train_op = tf.train.AdamOptimizer().minimize(_loss, global_step=_step)
# Init op initalizes variables to their default values
_init_op = tf.global_variables_initializer()

# The Saver handles saving all model parameters at checkpoints and
# restoring them later
_vars_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
_saver = tf.train.Saver(_vars_to_save)


def restore(session):
    latest_ckpt = tf.train.latest_checkpoint("./ckpts")
    if latest_ckpt is None:
        raise ValueError("Latest checkpoint does not exist")

    _saver.restore(session, latest_ckpt)
    console.info("Restored model from", latest_ckpt)


class LstmClassifier(Classifier):
    
    def __init__(self):
        pass

    def train(self,
        raw_inputs,
        true_labels,
        num_epochs=None,
        continue_previous=True,
        save_every_n_epochs=5,
        save_hook=None):
        """
        Args:
            `raw_inputs` a list of raw tweets. i.e, a list of strings.
            `true_labels` a list of integer labels corresponding to each raw input.
            `num_epochs` number of training epochs to run. If None, train forever.
            `continue_previous` if `True`, load params from latest checkpoint and
                continue training from there.
            `save_every_n_epochs`
            `save_hook` if given, a callable that will be called whenever a
                new checkpoint is saved
        """
        char_inputs = lstm_util.encode_raw_inputs(raw_inputs)
        train_data = lstm_util.make_batch(char_inputs, true_labels)

        train_feed = {
            _batch_size:   None,
            _inputs:       None,
            _labels:       None,
            _true_lengths: None
        }

        epochs_done = 0
        losses = deque(maxlen=10)
        minibatcher = lstm_util.Minibatcher(train_data)

        with tf.Session() as sess:
            try:
                restore(sess)
            except Exception as e:
                console.warn("Failed to restore from previous checkpoint:", e)
                console.warn("The model will be initialized from scratch.")
                sess.run(_init_op)

            while True:
                if num_epochs is not None and minibatcher.cur_epoch > num_epochs:
                    break

                batch = minibatcher.next(100)
                train_feed[_batch_size]   = len(batch.xs)
                train_feed[_inputs]       = batch.xs
                train_feed[_labels]       = batch.ys
                train_feed[_true_lengths] = batch.lengths

                [_, cur_loss] = sess.run([_train_op, _loss], train_feed)
                losses.append(cur_loss)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.log()
                    epochs_done += 1
                    if epochs_done % save_every_n_epochs == 0:
                        saved_path = _saver.save(sess, "./ckpts/lstm", global_step=_step)
                        console.log(console.colors.GREEN + console.colors.BRIGHT
                            + "\tCheckpoint saved to " + saved_path + console.colors.END)
                    console.log("\tAverage loss (last {} steps): {}".format(len(losses), np.mean(losses)))
                    if save_hook is not None:
                        save_hook()
                else:
                    # Print some stuff so we know it's making progress
                    label = "Global Step {}".format(sess.run(_step))
                    console.progress_bar(label, minibatcher.epoch_progress, 60)


    def predict(self, raw_inputs):
        with tf.Session() as sess:
            restore(sess)
            input_as_chars = lstm_util.encode_raw_inputs(raw_inputs)
            batch = lstm_util.make_batch(input_as_chars, labels=None)
            feed = {
                _batch_size:   len(batch.xs),
                _inputs:       batch.xs,
                _true_lengths: batch.lengths
            }
            predictions = sess.run(_labels_predicted, feed)
        return predictions
