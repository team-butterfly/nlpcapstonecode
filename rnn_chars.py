"""
A character-level LSTM in Tensorflow.
"""
import numpy as np
import tensorflow as tf


LOGGING = True
def log(*args):
    if LOGGING: print(*args)

CORPUS_LOC = "/homes/iws/brandv2/nlp/wiki/large.txt"

log("Loading corpus from", CORPUS_LOC)
with open(CORPUS_LOC) as f:
    corpus = np.fromiter(map(ord, f.read()), np.uint8)

NUM_WORDS   = len(corpus)
VOCAB_SIZE  = 256
EMBED_SIZE  = 256

SEQ_LEN     = 50
HIST_LEN    = SEQ_LEN-1
NUM_SAMPLES = NUM_WORDS-SEQ_LEN

LSTM_SIZE   = 128
RNN_LAYERS  = 1 # Currently, adding more layers results in garbage samples.

BATCH_SIZE  = 100
EPOCHS      = 10000
AVG_STEPS   = 100
SAMPLE_LEN  = 100
SAMPLE_TEMP = 0.5

log("Loaded corpus: vocab size is", VOCAB_SIZE,
    "max seq length is", SEQ_LEN)

def embeddings(xs):
    """
    Input:  an iterable of integer lexeme ids
    Return: a matrix of word embeddings with dims [num_words, embed_size]
    """
    n = len(xs)
    output = np.zeros([n, EMBED_SIZE], np.bool)
    # Slow version:
    # for i, lexid in enumerate(xs):
    #     output[i, lexid] = 1
    output[np.arange(n), xs] = True
    return output

def yield_all_pairs():
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

def run():
    graph = tf.Graph()
    with graph.as_default():
        batch_size = tf.placeholder(tf.int32)
        inputs     = tf.placeholder(tf.float32,
                                    [None, None, EMBED_SIZE],
                                    name="inputs")
        true_lengths = tf.placeholder(tf.int32, [None], name="true_lengths")
        labels = tf.placeholder(tf.int32,
                                [None], 
                                name="labels")

        temperature = tf.placeholder(tf.float32, [1], "temperature")

        cell = tf.contrib.rnn.LSTMCell(LSTM_SIZE, state_is_tuple=True)
        
        # Add multi-layeredness
        if RNN_LAYERS > 1:
            cell = tf.contrib.rnn.MultiRNNCell([cell] * RNN_LAYERS, state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)
        states, final_state_tuple = tf.nn.dynamic_rnn(
            cell,
            inputs,
            initial_state=initial_state,
            sequence_length=true_lengths,
            dtype=tf.float32)

        final_states = states[:, -1, :]
        unembed = dense(LSTM_SIZE, VOCAB_SIZE)
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
        for t in range(EPOCHS + 1):
            print(".", end="")
            batch_inputs, batch_labels = get_batch(BATCH_SIZE)
            train_feed = {
                batch_size: BATCH_SIZE,
                inputs: batch_inputs,
                true_lengths: np.repeat(HIST_LEN, BATCH_SIZE),
                labels: batch_labels
            }
            _, cur_loss = sess.run([train_step, loss], train_feed)
            avg_loss += cur_loss

            if t % AVG_STEPS != 0:
                continue

            # Every few steps, report average loss and sample a
            # "dream" sentence from the RNN

            avg_loss /= (AVG_STEPS if t > 0 else 1)
            print("Step", t)
            print("\tAverage loss:", avg_loss)
            
            def encode(st):
                return [ord(ch) for ch in st]
            
            def decode(lst):
                return "".join(chr(i) for i in lst)

            seed  = "In 1995 , a dog "
            dream = encode(seed)

            as_input = embeddings(dream)
            as_input = as_input[np.newaxis, :, :]
            sample_feed = {
                batch_size: 1,
                true_lengths: [len(dream)],
                inputs: as_input
            }
            cur_state = sess.run(final_state_tuple, sample_feed)

            for _ in range(SAMPLE_LEN):
                char = dream[-1]
                as_input = embeddings([char])
                as_input = as_input[np.newaxis, :, :] # Reshape to [batch_size, ...]
                sample_feed = {
                    batch_size: 1,
                    true_lengths: [1],
                    initial_state: cur_state,
                    temperature: [SAMPLE_TEMP],
                    inputs: as_input
                }
                [probs, cur_state] = sess.run([sample_outputs, final_state_tuple], sample_feed)
                if char == " ":
                    item = np.random.choice(VOCAB_SIZE, p=np.ravel(probs))
                else:
                    item = np.argmax(probs)
                dream.append(item)
            
            print("\tSample:", decode(dream))
