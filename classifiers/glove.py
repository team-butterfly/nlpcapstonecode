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
from .batch import Batch, Minibatcher, pad_to_max_len, lengths


def xavier(size_in, size_out):
    d = np.sqrt(6 / (size_in + size_out))
    return tf.random_uniform((size_in, size_out), minval=-d, maxval=d) 


class _GloveGraph():

    NUM_LABELS = len(Emotion)

    def __init__(self, vocab_size, embed_size, initial_embeddings):
        self.HIDDEN_SIZE = embed_size
        vocab_size += 1 # Add 1 for UNK

        console.info("Building Glove graph")
        self.root = tf.Graph()
        with self.root.as_default():
            # Model inputs
            self.batch_size   = tf.placeholder(tf.int32, name="batch_size")
            self.inputs       = tf.placeholder(tf.int32, [None, None], name="inputs")
            self.labels       = tf.placeholder(tf.int32, [None], name="labels")
            self.true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")

            self.train_embeddings = tf.constant(False, name="train_embeddings")
            self.use_dropout = tf.constant(False, name="use_dropout")
            self.keep_prob = tf.constant(0.5, name="keep_prob")

            self.unk_vec = tf.stop_gradient(tf.zeros([1, embed_size], tf.float32))
            self.embeddings = tf.Variable(
                tf.concat([initial_embeddings, self.unk_vec], axis=0),
                dtype=tf.float32,
                name="embeddings")

            # if train_embeddings is False, stop gradient backprop at the embedded inputs.
            self.inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.inputs, max_norm=1.0)
            self.inputs_embedded_frozen = tf.stop_gradient(self.inputs_embedded)
            self.inputs_embedded = tf.where(
                self.train_embeddings,
                self.inputs_embedded,
                self.inputs_embedded_frozen)

            # if self.use_dropout, then this is keep_prob, and otherwise it is 1.0
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

            self.concat_final_states = tf.concat([final_fw.h, final_bw.h], axis=1)

            # --------------------
            # Attention mechanism:
            #   let v_i = word embedding for i-th word
            #   let o = sum(forward lstm output, backward lstm output)
            #   let e_i = cos_sim(o, v_i) OR o*W*v_i
            #   let a = smooth(e) OR softmax(e)
            #   let x = sum(a_i * v_i)
            #   let y_hat = softmax(xW + b)

            def dot(a, b):
                return tf.reduce_sum(tf.multiply(a, b), axis=2)

            self.o = final_fw.h + final_bw.h 

            mask = tf.sequence_mask(self.true_lengths)
            float_mask = tf.cast(mask[:, :, tf.newaxis], tf.float32)

            self.e = dot(self.o[:, tf.newaxis, :], self.inputs_embedded * float_mask)
            self.e = self.e / tf.norm(self.o, axis=1, keep_dims=True)

            scores = tf.where(
                tf.sequence_mask(self.true_lengths),
                self.e,
                tf.ones_like(self.e) * -1E8)

            self.a = tf.nn.softmax(scores)
            self.x = tf.reduce_sum(tf.multiply(self.a[:, :, tf.newaxis], self.inputs_embedded), axis=1)

            self.x = self.x / tf.norm(self.x, axis=1, keep_dims=True)

            self.w = tf.Variable(xavier(self.HIDDEN_SIZE, self.NUM_LABELS), name="dense_weights")
            self.b = tf.Variable(tf.zeros(self.NUM_LABELS), name="dense_biases")
            self.logits = tf.nn.xw_plus_b(self.x, self.w, self.b)
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
            self.saver = tf.train.Saver(max_to_keep=5)

            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("loss", self.loss)
            self.merged = tf.summary.merge_all()




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

    def _tokens_to_word_ids(self, input_tokens, true_labels):
        console.time("tokens to word IDs")
        wordids = np.zeros(len(input_tokens), dtype=object)

        def lookup(word):
            word_lower = word.lower()
            return self._glove["word_index"].get(word_lower, self._glove["vocab_size"])

        for i, sentence in enumerate(input_tokens):
            wordids[i] = np.fromiter((lookup(w) for w in sentence), dtype=np.int, count=len(sentence))

        console.time_end("tokens to word IDs")
        return wordids

    def train(
        self,
        input_tokens,
        true_labels,
        logdir,
        num_epochs=None,
        continue_previous=True,
        save_every_n_epochs=1,
        eval_tokens=None,
        eval_labels=None,
        batch_size=512):

        """
        Args:
            `input_tokens` a list of tokenized sentences, i.e. a List[List[String]]
            `true_labels` a list of integer labels corresponding to each raw input.
            `logdir` path to save TensorBoard files
            `num_epochs` number of training epochs to run. If None, train forever.
            `continue_previous` if `True`, load params from latest checkpoint and
                continue training from there.
            `save_every_n_epochs` save a checkpoint at this interval, and also report evaluation metrics. 
            `eval_tokens` (optional) inputs for evaluation
            `eval_labels` (optional) labels for evaluation
            `batch_size` training batch size
        """

        train_data = self._tokens_to_word_ids(input_tokens, true_labels)
        true_labels = np.array(true_labels, dtype=np.int)

        eval_data = self._tokens_to_word_ids(eval_tokens, eval_labels)
        eval_labels = np.array(eval_labels, dtype=np.int)

        """
        def decode(ws):
            return [self._glove["index_word"][i] if i < len(self._glove["index_word"]) else "<UNK>"
                for i in ws]
        """

        # feed dict for training steps
        train_feed = {
            self._g.use_dropout: True,
            self._g.train_embeddings: True 
        }

        # feed dict for evaluating on the validation set
        eval_feed = {
            self._g.batch_size:   len(eval_data),
            self._g.inputs:       pad_to_max_len(eval_data),
            self._g.labels:       eval_labels,
            self._g.true_lengths: lengths(eval_data)
        }

        minibatcher = Minibatcher(Batch(train_data, true_labels, lengths(train_data)))

        with tf.Session(graph=self._g.root) as sess:
            try:
                self._restore(sess)
            except Exception as e:
                console.warn("Failed to restore from previous checkpoint:", e)
                console.warn("The model will be initialized from scratch.")
                sess.run(self._g.init_op)

            writer = tf.summary.FileWriter(logdir, self._g.root)

            while True:
                if num_epochs is not None and minibatcher.cur_epoch > num_epochs:
                    break

                batch = minibatcher.next(batch_size, pad_per_batch=True)
                train_feed[self._g.batch_size]   = len(batch.xs)
                train_feed[self._g.inputs]       = batch.xs
                train_feed[self._g.labels]       = batch.ys
                train_feed[self._g.true_lengths] = batch.lengths

                [_, cur_loss, step] = sess.run([self._g.train_op, self._g.loss, self._g.step], train_feed)
                if step % 100 == 0 and eval_tokens is not None and eval_feed is not None:
                    [summary] = sess.run([self._g.merged], eval_feed)
                    writer.add_summary(summary, step)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.info("")
                    if False and minibatcher.cur_epoch % save_every_n_epochs == 0:
                        saved_path = self._g.saver.save(sess, "ckpts/glove/noattn", global_step=self._g.step)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tCheckpoint saved to {}".format(datetime.now(), saved_path)
                            + console.colors.END)

                        """
                        [a, x, e] = sess.run([self._g.a, self._g.x, self._g.e], eval_feed)
                        len_ = batch.lengths[20]
                        sent = decode(batch.xs[20][:len_])
                        info = list(zip(sent, a[20][:len_], e[20][:len_]))
                        info.sort(key=lambda t: -t[1]) # Sort by attn. weight, decreasing
                        print(" ".join(sent))
                        print("label[20] =", batch.ys[20])
                        for rec in info[:5]:
                            print(rec)
                        """

                # Not a new epoch - print some stuff to report progress
                else:
                    label = "Global Step {} (Epoch {})".format(step, minibatcher.cur_epoch)
                    console.progress_bar(label, minibatcher.epoch_progress, 60)


    def predict(self, raw_inputs):
        return np.argmax(self.predict_soft(raw_inputs), axis=1)


    def predict_soft(self, input_tokens):
        inputs = self._tokens_to_word_ids(input_tokens, None)
        feed = {
            self._g.batch_size:   len(inputs),
            self._g.inputs:       inputs,
            self._g.true_lengths: lengths(inputs)
        }
        with tf.Session(graph=self._g.root) as sess:
            self._restore(sess)
            soft_labels = sess.run(self._g.softmax, feed)
        return soft_labels
