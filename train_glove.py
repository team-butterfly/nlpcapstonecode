"""
Runs a simple train/eval loop of the GloVe LSTM
"""
import argparse
import sys
from os.path import isfile
from utility import console

parser = argparse.ArgumentParser(description="Train and evaluate the GloVe LSTM model")
parser.add_argument("--save", type=int, nargs="?", default=1, help="Save model every N epochs")
parser.add_argument("--epochs", type=int, nargs="?", default=None, help="Number of epochs")
parser.add_argument("--logdir", type=str, required=True, help="Where to save Tensorboard logs")
args = parser.parse_args()

glove = "glove.dict.200d.pkl"

if not isfile(glove):
    console.warn("couldn't find glove file", glove)
    console.warn("you could download it with wget homes.cs.washington.edu/~brandv2/nlp/" + glove)
    sys.exit(1)

console.info("Will save every {} epochs for {} epochs".format(
    args.save,
    args.epochs if args.epochs is not None else "unlimited"))

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers import GloveClassifier

# Load data source
data_src = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5)

# Print info about the data distribution and MFC accuracy
mfc_class = np.argmax(np.bincount(data_src.train_labels))
console.info("train distribution", np.bincount(data_src.train_labels))
mfc_acc_train = np.equal(mfc_class, data_src.train_labels).mean()
mfc_acc_test  = np.equal(mfc_class, data_src.test_labels).mean()
console.info("train mfc", mfc_acc_train)
console.info("test mfc", mfc_acc_test)
console.info("Logdir:", args.logdir)

lstm = GloveClassifier()
lstm.train(
    data_src.train_inputs,
    data_src.train_labels,
    logdir=args.logdir,
    initial_embeddings_pkl=glove,
    save_every_n_epochs=args.save,
    num_epochs=args.epochs,
    eval_tokens=data_src.test_inputs,
    eval_labels=data_src.test_labels,
    continue_previous=False,
    batch_size=256)
