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

from utility import console, Emotion
from utility.strings import read_vocab
from data_source import TweetsDataSource
from .classifier import Classifier
from .batch import Batch, Minibatcher, pad_to_max_len, lengths


def plot_attention(sent, attn, rgb, title, path):
    import matplotlib; matplotlib.use("Agg") # Backend without interactive display
    import matplotlib.pyplot as plt
    assert len(sent) == len(attn)
    assert len(rgb) == 3

    fig, ax = plt.subplots()
    ax.axis("off")
    t = ax.transData
    renderer = ax.figure.canvas.get_renderer()
    rgba = np.append(rgb, 1.0) # Append alpha channel
    lo, hi = min(attn), max(attn)
    w, max_w = 0, 312
    bbox = {"fc": rgba, "ec": (0, 0, 0, 0), "boxstyle": "round"} # Word bounding box
    for s, a in zip(sent, attn):
        rgba[3] = (a - lo) / (hi - lo) 
        text = ax.text(0, 0.9, s, bbox=bbox, transform=t, size=12, fontname="Monospace")
        text.draw(renderer)
        ex = text.get_window_extent()
        if w > max_w:
            t = matplotlib.transforms.offset_copy(text._transform, x=-w, y=-ex.height*2, units="dots")
            w = 0
        else:
            dw = ex.width + 20
            t = matplotlib.transforms.offset_copy(text._transform, x=dw, units="dots")
            w += dw
    plt.title(title)
    plt.savefig(path, transparent=True)
    plt.close(fig)


def xavier(size_in, size_out):
    d = np.sqrt(6 / (size_in + size_out))
    return tf.random_uniform((size_in, size_out), minval=-d, maxval=d) 


class HParams():
    """
    Hyperparameters 'struct'
    """
    def __init__(self):
        self.learning_rate = 1e-3
        self.epsilon = 1e-8
        self.hidden_size = 200
        self.keep_prob_in = 0.5
        self.keep_prob_out = 0.5
        self.batch_size = 128
 
    def __str__(self):
        items = sorted([k + ": " + str(v) for k, v in self.__dict__.items()])
        return "\n".join(items)


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
            w_m = tf.Variable(xavier(hp.hidden_size*2, hp.hidden_size*2), tf.float32)
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

            self.w = tf.Variable(xavier(hp.hidden_size, self.NUM_LABELS), name="dense_weights")
            self.b = tf.Variable(tf.zeros(self.NUM_LABELS), name="dense_biases")

            # self.logits = tf.nn.xw_plus_b(self.x, self.w, self.b)
            self.logits = tf.nn.xw_plus_b(self.concat_final_states,
                tf.Variable(xavier(hp.hidden_size*2, self.NUM_LABELS)),
                tf.Variable(tf.zeros(self.NUM_LABELS)))
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
        console.time("Read vocab")
        self.index_word, self.word_index = read_vocab("vocab.glove.txt")
        console.time_end("Read vocab")
        self.vocab_size = len(self.index_word)
        
        self._sess = tf.Session()
        ckpt = pjoin("ckpts", "glove", name)
        console.info("Looking in", ckpt)
        latest = tf.train.latest_checkpoint(ckpt)
        saver = tf.train.import_meta_graph(latest + ".meta")
        saver.restore(self._sess, latest)

        self.inputs = tf.get_collection("inputs")[0]
        self.batch_size = tf.get_collection("batch_size")[0]
        self.true_lengths = tf.get_collection("true_lengths")[0]
        self.labels = tf.get_collection("labels")[0]
        self.softmax = tf.get_collection("softmax")[0]
        self.attention = tf.get_collection("attention")[0]


    def _unwordids(ids):
        return [self.index_word[i] if i < self.vocab_size else "[UNK]" for i in ids]


    def _make_feed(self, list_strs):
        def lookup(word):
            return self.word_index.get(word.lower(), self.vocab_size)

        tokens = [TweetsDataSource.tokenize(s) for s in list_strs]
        console.info("strs to tokens:", tokens)
        ids = np.zeros(len(tokens), dtype=object)
        for i, sent in enumerate(tokens):
            ids[i] = np.fromiter((lookup(w) for w in sent), dtype=np.int, count=len(sent))

        return {
            self.batch_size:   len(ids),
            self.inputs:       pad_to_max_len(ids),
            self.true_lengths: lengths(ids)
        }

    
    def predict(self, list_strs):
        feed = self._make_feed(list_strs)
        return self._sess.run(self.labels, feed)
        

    def predict_soft(self, list_strs):
        feed = self._make_feed(list_strs)
        return self._sess.run(self.softmax, feed)

    
    def predict_soft_with_attention(self, list_strs):
        feed = self._make_feed(list_strs)
        [soft_labels, attns] = self._sess.run([self.softmax, self.attention], feed)

        console.info("list_strs:", list_strs)
        console.info("feed:", feed)

        data = []
        for i in range(len(list_strs)):
            emos = {emo: soft_labels[i][emo.value] for emo in (Emotion.SADNESS, Emotion.ANGER, Emotion.JOY)}
            data.append((feed[self.inputs][i], emos, attns[i]))
        return data 


class GloveTraining():

    def __init__(self, name, hparams):
        self.hp = hparams
        try:
            self.logdir = pjoin("log", "glove", name)
            ckpt_dir = pjoin("ckpts", "glove", name)
            console.log("Trying logdir", self.logdir)
            console.log("Trying checkpoint dir", ckpt_dir)
            mkdir(self.logdir)
            mkdir(ckpt_dir)
            self.ckpt_file = pjoin(ckpt_dir, name)
        except FileExistsError as e:
            console.warn("Logging or checkpoint directory already exists; choose a unique name for this training instance.")
            raise e

        with open("glove.dict.200d.pkl", "rb") as f:
            self._glove = pickle.load(f)
        self._g = _GloveGraph(self._glove["vocab_size"], 200, self._glove["embeddings"], self.hp)

    def _tokens_to_word_ids(self, input_tokens):
        oov = tot = 0
        def lookup(word):
            nonlocal oov, tot
            word = word.lower()
            if word not in self._glove["word_index"]:
                oov += 1
            tot += 1
            return self._glove["word_index"].get(word, self._glove["vocab_size"])

        wordids = np.zeros(len(input_tokens), dtype=object)
        for i, sentence in enumerate(input_tokens):
            wordids[i] = np.fromiter((lookup(w) for w in sentence), dtype=np.int, count=len(sentence))
        console.info("GloveTraining: found {} OOV words ({:.2f}%)".format(oov, oov/tot))
        return wordids

    def _unwordids(ids):
        return [self._glove["index_word"][i] if i < len(self._glove["index_word"]) else "[UNK]" for i in ids]

    def run(
        self,
        data_source,
        *,
        num_epochs=None,
        save_every_n_epochs=1):
        """
        Args:
            `data_source` a DataSource implementation
            `num_epochs` number of training epochs to run. If None, train forever.
            `save_every_n_epochs` save a checkpoint at this interval, and also report evaluation metrics. 
            `batch_size` training batch size
        """

        ds = data_source
        train_data = self._tokens_to_word_ids(ds.train_inputs)
        true_labels = np.array(ds.train_labels, dtype=np.int)

        # Validation data
        eval_data = self._tokens_to_word_ids(ds.test_inputs)
        eval_labels = np.array(ds.test_labels, dtype=np.int)

        # Flags to enable or disable ...
        PLOTTING = False # Plotting attention samples.
        SAVING   = True  # Saving model parameters at checkpoint intervals (they're huge)
        LOGGING  = True  # Logging training progress to Tensorboard log file.

        # Feed dict for training steps
        train_feed = {
            self._g.use_dropout: True,
            self._g.train_embeddings: True 
        }

        # Feed dict for evaluating on the validation set
        eval_feed = {
            self._g.batch_size:   len(eval_data),
            self._g.inputs:       pad_to_max_len(eval_data),
            self._g.labels:       eval_labels,
            self._g.true_lengths: lengths(eval_data)
        }

        minibatcher = Minibatcher(Batch(train_data, true_labels, lengths(train_data)))

        if PLOTTING:
            # Choose some stratified sample of the evaluation data to visualize attention weights.
            SAMPLES_PER_CLASS = 4
            rand = np.random.RandomState(12) # Use fixed seed for reproducibility.
            indices = np.arange(len(eval_data))
            pools = [indices[eval_labels == em.value] for em in (Emotion.ANGER, Emotion.SADNESS, Emotion.JOY)]
            plot_idx = np.concatenate([rand.choice(pool, SAMPLES_PER_CLASS, replace=False) for pool in pools])
            del pools
            del indices
            plot_inputs = eval_data[plot_idx]
            plot_labels = eval_labels[plot_idx]
            plot_words = [self._unwordids(wordids) for wordids in plot_inputs]
            plot_colors = {
                Emotion.ANGER:   (1.000, 0.129, 0.345), # Red
                Emotion.SADNESS: (0.231, 0.639, 0.988), # Blue
                Emotion.JOY:     (0.671, 0.847, 0) # Light green
            }

            plot_feed = {
                self._g.batch_size:   len(plot_inputs),
                self._g.inputs:       pad_to_max_len(plot_inputs),
                self._g.labels:       plot_labels,
                self._g.true_lengths: lengths(plot_inputs)
            }
        # end if PLOTTING

        with tf.Session(graph=self._g.root) as sess:
            sess.run(self._g.init_op)

            if LOGGING:
                writer = tf.summary.FileWriter(self.logdir)
                console.info("Logging to", self.logdir)
            if SAVING:
                console.info("Saving checkpoints to", self.ckpt_file)

            while True:
                if num_epochs is not None and minibatcher.cur_epoch > num_epochs:
                    break

                batch = minibatcher.next(self.hp.batch_size, pad_per_batch=True)
                train_feed[self._g.batch_size]   = len(batch.xs)
                train_feed[self._g.inputs]       = batch.xs
                train_feed[self._g.labels]       = batch.ys
                train_feed[self._g.true_lengths] = batch.lengths

                [_, cur_loss, step] = sess.run([self._g.train_op, self._g.loss, self._g.step], train_feed)

                if LOGGING and (step % 100 == 0 or minibatcher.is_new_epoch):
                    [summary] = sess.run([self._g.merged], eval_feed)
                    writer.add_summary(summary, step)

                # At end of each epoch, maybe save and report some metrics
                if minibatcher.is_new_epoch:
                    console.info("")
                    valid_acc = np.equal(sess.run(self._g.labels_predicted, eval_feed), eval_labels).mean()
                    console.info("Validation accuracy:", valid_acc)
                    if SAVING and (minibatcher.cur_epoch % save_every_n_epochs == 0):
                        saved_path = self._g.saver.save(sess, self.ckpt_file, global_step=self._g.step, write_meta_graph=True)
                        console.log(
                            console.colors.GREEN + console.colors.BRIGHT
                            + "{}\tCheckpoint saved to {}".format(datetime.now(), saved_path)
                            + console.colors.END)

                    # Plot sample attentions.
                    if PLOTTING:
                        [attns, attn_preds] = sess.run([self._g.a, self._g.labels_predicted], plot_feed)
                        for i in range(len(plot_words)):
                            em = Emotion(plot_labels[i]) # The true label
                            em_pred = Emotion(attn_preds[i]) # Predicted label
                            title = "True label '{}', Predicted '{}'".format(em.name, em_pred.name)
                            fname = "plots/attn/epoch{:02d}_{:02d}".format(minibatcher.cur_epoch-1, i)
                            fname_plot = fname + ".png"
                            
                            attn_clip = attns[i][:len(plot_words[i])].tolist()
                            plot_attention(plot_words[i], attn_clip, plot_colors[em], title, fname_plot)
                            console.info("Saved plot to", fname_plot)
                            
                            obj = {
                                "words": plot_words[i],
                                "attention": attn_clip,
                                "true_label": em.name,
                                "pred_label": em_pred.name
                            }
                            fname_obj = fname + ".json"
                            json.dump(obj, open(fname_obj, "w"))
                            console.info("saved JSON to", fname_obj)
                    # end if PLOTTING
                # end if minibatcher.is_new_epoch
                # Not a new epoch - print some stuff to report progress
                else:
                    label = "Global Step {} (Epoch {})".format(step, minibatcher.cur_epoch)
                    console.progress_bar(label, minibatcher.epoch_progress, 60)
