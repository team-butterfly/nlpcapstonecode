"""
Defines a word-level LSTM with GloVe embeddings
"""
from datetime import datetime
from collections import deque, defaultdict
import pickle
import numpy as np
import tensorflow as tf

from utility import console, Emotion
from .classifier import Classifier
from .batch import Batch, Minibatcher


def xavier(size_in, size_out):
    d = np.sqrt(6 / (size_in + size_out))
    return tf.random_uniform((size_in, size_out), minval=-d, maxval=d) 


class _GloveGraph():

    HIDDEN_SIZE = 128
    NUM_LABELS = len(Emotion)

    def __init__(self, vocab_size, embed_size, initial_embeddings):
        console.info("Building Glove graph")
        self.root = tf.Graph()
        with self.root.as_default():
            # Model inputs
            self.batch_size   = tf.placeholder(tf.int32, name="batch_size")
            self.inputs       = tf.placeholder(tf.int32, [None, None], name="inputs")
            self.labels       = tf.placeholder(tf.int32, [None], name="labels")
            self.true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")

            self.train_embeddings = tf.constant(False, name="train_embeddings")

            # If true, dropout is applied to LSTM cell inputs and outputs.
            self.use_dropout = tf.constant(False, name="use_dropout")
            self.keep_prob = tf.constant(0.5, name="keep_prob")

            with tf.device("/cpu:0"):
                self.embeddings = tf.Variable(
                    initial_value=initial_embeddings,
                    dtype=tf.float32,
                    name="embeddings")
                self.inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.inputs, max_norm=1.0)
                # if train_embeddings is true, stop gradient backprop at the embedded inputs.
                self.inputs_embedded = tf.where(
                    self.train_embeddings,
                    self.inputs_embedded,
                    tf.stop_gradient(self.inputs_embedded))

            # if self.use_dropout, then _keep_prob else 1.0
            self.keep_prob_conditional = tf.where(
                self.use_dropout, 
                self.keep_prob,
                tf.constant(1.0))

            def make_cell(size):
                return tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.LSTMCell(size, state_is_tuple=True),
                    input_keep_prob=self.keep_prob_conditional,
                    output_keep_prob=self.keep_prob_conditional)
            
            # Bi-LSTM here
            self.cell_fw = make_cell(self.HIDDEN_SIZE)
            self.cell_bw = make_cell(self.HIDDEN_SIZE)

            self.outputs, (final_fw, final_bw) = tf.nn.bidirectional_dynamic_rnn(
                self.cell_fw,
                self.cell_bw,
                self.inputs_embedded,
                initial_state_fw=self.cell_fw.zero_state(self.batch_size, tf.float32),
                initial_state_bw=self.cell_bw.zero_state(self.batch_size, tf.float32),
                sequence_length=self.true_lengths,
                dtype=tf.float32)

            self.final_state = tf.concat([final_fw.h, final_bw.h], axis=1)

            # Weights and biases for the fully-connected layer that
            # projects the final LSTM state down to the size of the label space
            self.w = tf.Variable(xavier(self.HIDDEN_SIZE*2, self.NUM_LABELS), name="dense_weights")
            self.b = tf.Variable(tf.zeros(self.NUM_LABELS), name="dense_biases")

            self.logits = tf.nn.xw_plus_b(self.final_state, self.w, self.b)
            self.softmax = tf.nn.softmax(self.logits)

            # Must cast tf.argmax to int32 because it returns int64
            self.labels_predicted = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)

            # Loss function is mean cross-entropy
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels,
                logits=self.logits)) # + 0.01 * tf.nn.l2_loss(self.embeddings)

            # Global step counter (incremented by the AdamOptimizer on each gradient update)
            self.step = tf.Variable(0, name="global_step", trainable=False)
            # When executed, train op runs one step of gradient descent.
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.step)
            # Init op initalizes variables to their default values
            self.init_op = tf.global_variables_initializer()

            # The Saver handles saving all model parameters at checkpoints and
            # restoring them later
            self.vars_saved = tf.trainable_variables() + [self.step]
            self.saver = tf.train.Saver(self.vars_saved, max_to_keep=10)


class GloveClassifier(Classifier):
    
    def __init__(self, embeddings_pkl):
        console.info("Loading vectors from", embeddings_pkl)
        console.time("GloveVectors init")
        with open(embeddings_pkl, "rb") as f:
            self._glove = pickle.load(f)
        console.time_end("GloveVectors init")
        self._g = _GloveGraph(
            self._glove["vocab_size"],
            self._glove["dimension"],
            self._glove["embeddings"])

    def _restore(self, session):
        latest_ckpt = tf.train.latest_checkpoint("./ckpts/glove")
        if latest_ckpt is None:
            raise ValueError("Latest checkpoint does not exist")

        self._g.saver.restore(session, latest_ckpt)
        console.info("Restored model from", latest_ckpt)

    def _batch_from_tokens(self, input_tokens, true_labels):
        console.time("batch from tokens")
        maxlen = max(len(sentence) for sentence in input_tokens)
        wordids = np.zeros([len(input_tokens), maxlen], np.int32)
        lengths = np.zeros(len(input_tokens), np.int32)

        for i, sentence in enumerate(input_tokens):
            lengths[i] = len(sentence)
            for j, token in enumerate(sentence):
                token_lower = token.lower() # GloVe vocabulary is uncased
                wordids[i, j] = self._glove["word_index"].get(token_lower, 0)

        console.time_end("batch from tokens")
        return Batch(
            xs=wordids,
            ys=np.array(true_labels, np.int32) if true_labels is not None else None,
            lengths=lengths)

    def train(self,
        input_tokens,
        true_labels,
        num_epochs=None,
        continue_previous=True,
        save_every_n_epochs=5,
        eval_tokens=None,
        eval_labels=None):
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

        # feed dict for evaluating on the validation set
        eval_data = self._batch_from_tokens(eval_tokens, eval_labels)
        eval_feed = {
            self._g.batch_size:   len(eval_data.xs),
            self._g.inputs:       eval_data.xs,
            self._g.true_lengths: eval_data.lengths
        }

        # feed dict for training steps
        train_data = self._batch_from_tokens(input_tokens, true_labels)
        train_feed = {
            self._g.use_dropout:  True
        }

        # a feed dict for evaluating on the training set (to detect overfitting)
        train_eval_feed = {
            self._g.batch_size:   len(train_data.xs),
            self._g.inputs:       train_data.xs,
            self._g.true_lengths: train_data.lengths
        }

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

                batch = minibatcher.next(256)
                train_feed[self._g.train_embeddings] = False # minibatcher.cur_epoch < 8
                train_feed[self._g.batch_size]   = len(batch.xs)
                train_feed[self._g.inputs]       = batch.xs
                train_feed[self._g.labels]       = batch.ys
                train_feed[self._g.true_lengths] = batch.lengths

                [_, cur_loss, step] = sess.run([self._g.train_op, self._g.loss, self._g.step], train_feed)
                losses.append(cur_loss)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.info("\tAverage loss (last {} steps): {:.4f}".format(len(losses), np.mean(losses)))
                    if minibatcher.cur_epoch % save_every_n_epochs == 0:
                        # don't save because it's massive
                        """
                        saved_path = self._g.saver.save(sess, "./ckpts/glove/glove_saved", global_step=self._g.step)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tCheckpoint saved to {}".format(datetime.now(), saved_path)
                            + console.colors.END)
                        """

                        eval_pred = sess.run(self._g.labels_predicted, eval_feed)
                        train_pred = sess.run(self._g.labels_predicted, train_eval_feed)
                        train_acc = np.equal(train_pred, train_data.ys).mean()
                        eval_acc = np.equal(eval_pred, eval_labels).mean()
                        console.log("train accuracy: {:.5f}".format(train_acc))
                        console.log("eval  accuracy: {:.5f}".format(eval_acc))
                else:
                    # Print some stuff so we know it's making progress
                    label = "Global Step {} (Epoch {})".format(step, minibatcher.cur_epoch)
                    console.progress_bar(label, minibatcher.epoch_progress, 60)


    def predict(self, raw_inputs):
        return np.argmax(self.predict_soft(raw_inputs), axis=1)


    def predict_soft(self, input_tokens):
        batch = self._batch_from_tokens(input_tokens, None)
        feed = {
            self._g.use_dropout:  False,
            self._g.batch_size:   len(batch.xs),
            self._g.inputs:       batch.xs,
            self._g.true_lengths: batch.lengths
        }
        with tf.Session(graph=self._g.root) as sess:
            self._restore(sess)
            soft_labels = sess.run(self._g.softmax, feed)
        return soft_labels
