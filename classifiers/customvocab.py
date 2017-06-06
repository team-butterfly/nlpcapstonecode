"""
Defines a word-level LSTM with custom embeddings
"""
from datetime import datetime
from os.path import join as pjoin
from os import mkdir
from shutil import rmtree
import pickle
import numpy as np
import tensorflow as tf

import classifiers.utility as util

from utility import console, Emotion
from utility.strings import read_vocab, write_vocab, StringStore
from data_source import TweetsDataSource
from data_source.tokenize import tokenize_tweet
from .classifier import Classifier


class _CustomVocabGraph():

    NUM_LABELS = len(Emotion)

    def __init__(self, hparams, init_embeddings=None):
        console.info("Building Glove graph")

        hp = hparams
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

            if init_embeddings is None:
                init_embeddings = tf.random_uniform([hp.vocab_size, hp.embed_size], -1, 1) 

            self.embeddings = tf.Variable(init_embeddings, dtype=tf.float32, name="embeddings")

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

            # Attention mechanism
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

            scores = tf.where(mask, self.e, tf.ones_like(self.e) * -1E8)

            self.a = tf.nn.softmax(scores)
            self.x = tf.reduce_sum(tf.multiply(self.a[:, :, tf.newaxis], attn_target), axis=1)

            self.x = self.x / tf.norm(self.x, axis=1, keep_dims=True)

            self.w = tf.Variable(util.xavier(hp.hidden_size*2, self.NUM_LABELS), name="dense_weights")
            self.b = tf.Variable(tf.zeros(self.NUM_LABELS), name="dense_biases")

            # Use these logits to enable attention:
            self.logits_a = tf.nn.xw_plus_b(self.x, self.w, self.b)
            
            # Use these logits to bypass attention:
            self.logits_b = tf.nn.xw_plus_b(self.concat_final_states,
                tf.Variable(util.xavier(hp.hidden_size*2, self.NUM_LABELS)),
                tf.Variable(tf.zeros(self.NUM_LABELS)))
            
            # Direct sum of attention and non-attention vectors
            self.logits_c = tf.nn.xw_plus_b(self.concat_final_states + self.x,
                tf.Variable(util.xavier(hp.hidden_size*2, self.NUM_LABELS)),
                tf.Variable(tf.zeros(self.NUM_LABELS)))

            # Weighted sum of attention and non-attention vectors
            weighted = lambda t: t * tf.Variable(tf.random_uniform([hp.hidden_size*2]))
            self.logits_d = tf.nn.xw_plus_b(
                weighted(self.concat_final_states) + weighted(self.x),
                tf.Variable(util.xavier(hp.hidden_size*2, self.NUM_LABELS)),
                tf.Variable(tf.zeros(self.NUM_LABELS)))

            # Attention with extra hidden layer
            self.logits_e = tf.nn.xw_plus_b(
                tf.nn.xw_plus_b(
                    self.x,
                    tf.Variable(util.xavier(hp.hidden_size*2, 64)),
                    tf.Variable(tf.zeros(64))),
                tf.Variable(util.xavier(64, self.NUM_LABELS)),
                tf.Variable(tf.zeros(self.NUM_LABELS)))

            self.logits = self.logits_a
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


class CustomVocabClassifier(Classifier):

    def __init__(self, name):
        self.index_word, self.word_index = read_vocab("ckpts/customvocab/{}/{}.vocab".format(name, name))
        self.vocab_size = len(self.index_word)

        ckpt = pjoin("ckpts", "customvocab", name)
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


class CustomVocabTraining(util.TrainingSession):

    def __init__(self, run_name, hparams):
        super().__init__("customvocab", run_name)
        self.hparams = hparams


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
        self.vocab = StringStore(map(tokenize_tweet, ds.train_raw_inputs), self.hparams.vocab_size)
        initial_embeddings = None

        """
        # If word is in GloVe twitter, initialize with the GloVe embedding.
        # Otherwise, initialize with random noise.
        n_found = 0
        initial_embeddings = np.random.uniform(-1, 1, [self.hparams.vocab_size, self.hparams.embed_size])
        with open("glove.dict.200d.pkl", "rb") as f:
            glove = pickle.load(f)
        for i, word in enumerate(self.vocab):
            if word in glove["word_index"]:
                idx = glove["word_index"][word]
                glove_vec = glove["embeddings"][idx]
                assert self.vocab.id2word(i) == word, "self.vocab[{}] got {}, want {}".format(i, self.vocab.id2word(i), word) 
                initial_embeddings[i] = glove_vec
                n_found += 1
        console.info("{}/{} words found in GloVe".format(n_found, self.hparams.vocab_size))
        """

        graph = _CustomVocabGraph(self.hparams, initial_embeddings)

        def sent_to_ids(corpus):
            return np.array([
                np.fromiter(map(self.vocab.word2id, tokenize_tweet(sent)), dtype=int) for sent in corpus],
                dtype=np.ndarray)
        train_inputs = sent_to_ids(ds.train_raw_inputs) 
        true_labels = np.array(ds.train_labels, dtype=np.int)

        # Validation data
        eval_inputs = sent_to_ids(ds.test_raw_inputs)
        eval_labels = np.array(ds.test_labels, dtype=np.int)

        # Feed dict for training steps
        train_feed = {
            graph.use_dropout: True,
            graph.train_embeddings: True,
            graph.batch_size: None, 
            graph.inputs: None,
            graph.labels: None, 
            graph.true_lengths: None
        }

        # Feed dict for evaluating on the validation set
        eval_feed = {
            graph.batch_size:   len(eval_inputs),
            graph.inputs:       util.pad_to_max_len(eval_inputs),
            graph.labels:       eval_labels,
            graph.true_lengths: util.lengths(eval_inputs)
        }

        minibatcher = util.Minibatcher(util.Batch(train_inputs, true_labels, util.lengths(train_inputs)))

        if progress_interval is not None:
            next_progress = 0.0    

        with tf.Session(graph=graph.root) as sess:
            sess.run(graph.init_op)

            writer = tf.summary.FileWriter(self.logdir, flush_secs=60)

            while True:
                if num_epochs is not None and minibatcher.cur_epoch > num_epochs:
                    break

                batch = minibatcher.next(self.hparams.batch_size, pad_per_batch=True)
                train_feed[graph.batch_size]   = len(batch.xs)
                train_feed[graph.inputs]       = batch.xs
                train_feed[graph.labels]       = batch.ys
                train_feed[graph.true_lengths] = batch.lengths

                [_, cur_loss, step] = sess.run([graph.train_op, graph.loss, graph.step], train_feed)

                # Log validation accuracy to Tensorboard file
                if step > 0 and step % 100 == 0:
                    summary = sess.run(graph.merged, eval_feed)
                    writer.add_summary(summary, step)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.info("")

                    if progress_interval is not None:
                        next_progress = 0.0    

                    if save_interval is not None and minibatcher.cur_epoch % save_interval == 0:
                        saved_path = graph.saver.save(sess, self.ckpt_file, global_step=graph.step, write_meta_graph=True)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tCheckpoint saved to {}".format(datetime.now(), saved_path)
                            + console.colors.END)

                        write_vocab(self.vocab.vocab(), self.vocab_file)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tVocab saved to {}".format(datetime.now(), self.vocab_file)
                            + console.colors.END)

                    if eval_interval is not None and minibatcher.cur_epoch % eval_interval == 0:
                        eval_pred = sess.run(graph.labels_predicted, eval_feed)
                        eval_acc = np.equal(eval_pred, eval_labels).mean()
                        console.log("eval accuracy: {:.5f}".format(eval_acc))

                # Not a new epoch - print some stuff to report progress
                elif progress_interval is not None and minibatcher.epoch_progress >= next_progress:
                    label = "Global Step {} (Epoch {}) Loss {:.5f}".format(step, minibatcher.cur_epoch, cur_loss)
                    console.progress_bar(label, minibatcher.epoch_progress, 60)
                    next_progress += progress_interval
