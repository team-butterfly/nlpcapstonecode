"""
Defines a word-level LSTM with GloVe embeddings
"""
from datetime import datetime
from collections import namedtuple
from os.path import join as pjoin
from os.path import isdir 
from os import mkdir
import json
import pickle
import numpy as np
import tensorflow as tf

import classifiers.utility as util

from utility import console, Emotion
from utility.strings import read_vocab
from data_source import TweetsDataSource
from .classifier import Classifier


class _GloveGraph():

    NUM_LABELS = len(Emotion)

    def __init__(self, vocab_size, embed_size, initial_embeddings, hparams):
        console.info("Building Glove graph")

        hp = hparams
        vocab_size += 1 # Add 1 for UNK
        self.root = tf.Graph()

        with self.root.as_default():
            # Model inputs
            self.batch_size   = tf.placeholder(tf.int32, name="batch_size")
            self.inputs       = tf.placeholder(tf.int32, [None, None], name="inputs")
            self.labels       = tf.placeholder(tf.int32, [None], name="labels")
            self.true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")

            # Constants and variables
            self.train_embeddings = tf.constant(False, name="train_embeddings")
            self.use_dropout      = tf.constant(False, name="use_dropout")
            ONE = tf.constant(1.0)
            self.keep_prob_in = tf.where(self.use_dropout, tf.constant(hp.keep_prob_in), ONE)
            self.keep_prob_out = tf.where(self.use_dropout, tf.constant(hp.keep_prob_out), ONE)

            self.unk_vec = tf.stop_gradient(tf.zeros([1, embed_size], tf.float32))
            self.embeddings = tf.Variable(
                tf.concat([initial_embeddings, self.unk_vec], axis=0),
                dtype=tf.float32,
                name="embeddings")

            # if train_embeddings is False, stop gradient backprop at the embedded inputs.
            self.inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.inputs, max_norm=1.0)
            self.inputs_embedded = tf.where(
                self.train_embeddings,
                self.inputs_embedded,
                tf.stop_gradient(self.inputs_embedded))

            def make_cell(size):
                return tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.LSTMCell(size, state_is_tuple=True),
                    input_keep_prob=self.keep_prob_in,
                    output_keep_prob=self.keep_prob_out,
                    seed=4)
            
            # Bi-LSTM here
            self.cell_fw = make_cell(hp.hidden_size)
            self.cell_bw = make_cell(hp.hidden_size)

            (outputs_fw, outputs_bw), (final_fw, final_bw) = tf.nn.bidirectional_dynamic_rnn(
                self.cell_fw,
                self.cell_bw,
                self.inputs_embedded,
                initial_state_fw=self.cell_fw.zero_state(self.batch_size, tf.float32),
                initial_state_bw=self.cell_bw.zero_state(self.batch_size, tf.float32),
                sequence_length=self.true_lengths,
                dtype=tf.float32)

            self.concat_final_states = tf.concat([final_fw.h, final_bw.h], axis=1)

            # --------------------
            # Attention mechanism:
            #   let v_i = word embedding for i-th word
            #   let o = sum(forward lstm output, backward lstm output) or some other combination of them
            #   let e_i = cos_sim(o, v_i) OR o*W*v_i
            #   let a = normalized scores, i.e. smooth(e) OR softmax(e)
            #   let x = weighted average, i.e. sum(a_i * v_i)
            #   let y_hat = softmax(xW + b)

            o = self.concat_final_states 

            # attn_target = self.inputs_embedded # Attention on input embeddings
            attn_target = tf.concat([outputs_fw, outputs_bw], axis=2) # Attention on outputs
            mask = tf.sequence_mask(self.true_lengths)
            float_mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)

            # Bilinear parameterization with a (square) weights matrix
            w_m = tf.Variable(util.xavier(hp.hidden_size*2, hp.hidden_size*2), tf.float32)
            self.e = tf.reduce_sum(
                tf.matmul(o, w_m)[:, tf.newaxis, :] # broadcast across all time steps
                    * (attn_target * float_mask),
                axis=2)

            # Cosine similarity
            # self.e = tf.reduce_sum(o[:, tf.newaxis, :] * (self.inputs_embedded * float_mask), axis=2)
            # self.e = self.e / tf.norm(o, axis=1, keep_dims=True)

            scores = tf.where(mask & tf.not_equal(self.inputs, vocab_size-1), self.e, tf.ones_like(self.e) * -1E8)

            self.a = tf.nn.softmax(scores)
            self.x = tf.reduce_sum(tf.multiply(self.a[:, :, tf.newaxis], attn_target), axis=1)

            self.x = self.x / tf.norm(self.x, axis=1, keep_dims=True)

            self.w = tf.Variable(util.xavier(hp.hidden_size*2, self.NUM_LABELS), name="dense_weights")
            self.b = tf.Variable(tf.zeros(self.NUM_LABELS), name="dense_biases")

            # Use these logits to enable attention:
            self.logits = tf.nn.xw_plus_b(self.x, self.w, self.b)
            
            # Use these logits to bypass attention:
            # self.logits = tf.nn.xw_plus_b(self.concat_final_states,
            #     tf.Variable(util.xavier(hp.hidden_size*2, self.NUM_LABELS)),
            #     tf.Variable(tf.zeros(self.NUM_LABELS)))
            self.softmax = tf.nn.softmax(self.logits)

            # Must cast tf.argmax to int32 because it returns int64
            self.labels_predicted = tf.cast(tf.argmax(self.softmax, axis=1), tf.int32)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.labels_predicted, self.labels), tf.float32))

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
            self.saver = tf.train.Saver(max_to_keep=1)

            # Exported for reloading in the GloveClassifier
            tf.add_to_collection("inputs", self.inputs) 
            tf.add_to_collection("true_lengths", self.true_lengths)
            tf.add_to_collection("batch_size", self.batch_size)
            tf.add_to_collection("labels", self.labels_predicted) # Predicted labels
            tf.add_to_collection("softmax", self.softmax)
            tf.add_to_collection("attention", self.a) # Attention weights

            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("loss", self.loss)
            self.merged = tf.summary.merge_all()


# The "Classifier" ABC needs some re-thinking because having "train" and "predict" in the same class
# makes no sense for a model that must be trained in advance, saved to disk, and loaded to do predictions.
class GloveClassifier(Classifier):

    def __init__(self, name):
        self.index_word, self.word_index = read_vocab("vocab.glove.txt")
        self.vocab_size = len(self.index_word)

        ckpt = pjoin("ckpts", "glove", name)
        console.info("Looking in", ckpt)
        latest = tf.train.latest_checkpoint(ckpt)
        if latest is None:
            raise ValueError("No checkpoint found in {}".format(ckpt))

        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(latest + ".meta")
            saver.restore(self._sess, latest)

            self.inputs = tf.get_collection("inputs")[0]
            self.batch_size = tf.get_collection("batch_size")[0]
            self.true_lengths = tf.get_collection("true_lengths")[0]
            self.labels = tf.get_collection("labels")[0]
            self.softmax = tf.get_collection("softmax")[0]
            self.attention = tf.get_collection("attention")[0]


    def _unwordids(self, ids):
        return [self.index_word[i] if i < self.vocab_size else "[UNK]" for i in ids]


    def _make_feed(self, tokens):
        def lookup(word):
            return self.word_index.get(word.lower(), self.vocab_size)

        ids = np.zeros(len(tokens), dtype=object)
        for i, sent in enumerate(tokens):
            ids[i] = np.fromiter((lookup(w) for w in sent), dtype=np.int, count=len(sent))

        return {
            self.batch_size:   len(ids),
            self.inputs:       util.pad_to_max_len(ids),
            self.true_lengths: util.lengths(ids)
        }

    
    def predict(self, tokens):
        feed = self._make_feed(tokens)
        return self._sess.run(self.labels, feed)
        

    def predict_soft(self, tokens):
        feed = self._make_feed(tokens)
        return self._sess.run(self.softmax, feed)

    
    def predict_soft_with_attention(self, tokens):
        feed = self._make_feed(tokens)
        [soft_labels, attns] = self._sess.run([self.softmax, self.attention], feed)

        data = []
        for i in range(len(tokens)):
            emos = {emo: soft_labels[i][emo.value] for emo in (Emotion.SADNESS, Emotion.ANGER, Emotion.JOY)}
            num_words = feed[self.true_lengths][i]
            tokens = feed[self.inputs][i][:num_words]
            tokens = self._unwordids(tokens)
            data.append((tokens, emos, attns[i][:num_words]))
        return data 

    def close(self):
        self._sess.close()


class GloveTraining(util.TrainingSession):

    def __init__(self, run_name, hparams):
        super().__init__("glove", run_name)

        self.hp = hparams
        with open("glove.dict.200d.pkl", "rb") as f:
            self._glove = pickle.load(f)
        self._g = _GloveGraph(self._glove["vocab_size"], 200, self._glove["embeddings"], self.hp)

    def _tokens_to_word_ids(self, input_tokens):
        def lookup(word):
            word = word.lower()
            return self._glove["word_index"].get(word, self._glove["vocab_size"])

        wordids = np.zeros(len(input_tokens), dtype=object)
        for i, sentence in enumerate(input_tokens):
            wordids[i] = np.fromiter((lookup(w) for w in sentence), dtype=np.int, count=len(sentence))
        return wordids

    def _unwordids(self, ids):
        return [self._glove["index_word"][i] if i < len(self._glove["index_word"]) else "[UNK]" for i in ids]

    def run(
        self,
        data_source,
        *,
        num_epochs=None,
        save_interval=1,
        eval_interval=None,
        progress_interval=0.01):
        """
        Args:
            `data_source` a DataSource implementation
            `num_epochs` number of training epochs to run. If None, train forever.
            `save_interval`: positive integer (default: 1), or None
                if not None, save a checkpoint and report evaluation metrics
                every save_interval epochs
            `eval_interval`: positive integer, or None (default)
                if not None, write eval accuracy to console every eval_interval
                epochs
            `progress_interval`: float between 0 and 1 (default: 0.01), or None
                if not None, write a progress bar to the console at most
                1/progress_interval times per epoch (e.g. if set to 0.1, should
                display ~10 progress bars)
        """

        ds = data_source
        train_data = self._tokens_to_word_ids(ds.train_inputs)
        true_labels = np.array(ds.train_labels, dtype=np.int)

        # Validation data
        eval_data = self._tokens_to_word_ids(ds.test_inputs)
        eval_labels = np.array(ds.test_labels, dtype=np.int)

        # Feed dict for training steps
        train_feed = {
            self._g.use_dropout: True,
            self._g.train_embeddings: True
        }

        # Feed dict for evaluating on the validation set
        eval_feed = {
            self._g.batch_size:   len(eval_data),
            self._g.inputs:       util.pad_to_max_len(eval_data),
            self._g.labels:       eval_labels,
            self._g.true_lengths: util.lengths(eval_data)
        }

        minibatcher = util.Minibatcher(util.Batch(train_data, true_labels, util.lengths(train_data)))


        if progress_interval is not None:
            next_progress = 0.0    

        with tf.Session(graph=self._g.root) as sess:
            sess.run(self._g.init_op)

            writer = tf.summary.FileWriter(self.logdir)

            while True:
                if num_epochs is not None and minibatcher.cur_epoch > num_epochs:
                    break

                batch = minibatcher.next(self.hp.batch_size, pad_per_batch=True)
                train_feed[self._g.batch_size]   = len(batch.xs)
                train_feed[self._g.inputs]       = batch.xs
                train_feed[self._g.labels]       = batch.ys
                train_feed[self._g.true_lengths] = batch.lengths

                [_, cur_loss, step] = sess.run([self._g.train_op, self._g.loss, self._g.step], train_feed)

                # Log validation accuracy to Tensorboard file
                if step > 0 and step % 100 == 0:
                    summary = sess.run(self._g.merged, eval_feed)
                    writer.add_summary(summary, step)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.info("")

                    if progress_interval is not None:
                        next_progress = 0.0    

                    if save_interval is not None and minibatcher.cur_epoch % save_interval == 0:
                        saved_path = self._g.saver.save(sess, self.ckpt_file, global_step=self._g.step, write_meta_graph=True)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tCheckpoint saved to {}".format(datetime.now(), saved_path)
                            + console.colors.END)

                    if eval_interval is not None and minibatcher.cur_epoch % eval_interval == 0:
                        eval_pred = sess.run(self._g.labels_predicted, eval_feed)
                        eval_acc = np.equal(eval_pred, eval_labels).mean()
                        console.log("eval accuracy: {:.5f}".format(eval_acc))

                # Not a new epoch - print some stuff to report progress
                elif progress_interval is not None and minibatcher.epoch_progress >= next_progress:
                    label = "Global Step {} (Epoch {})".format(step, minibatcher.cur_epoch)
                    console.progress_bar(label, minibatcher.epoch_progress, 60)
                    next_progress += progress_interval
