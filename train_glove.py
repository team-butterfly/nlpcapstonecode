"""
Runs a simple train/eval loop of the GloVe LSTM
"""
import argparse
from utility import console

parser = argparse.ArgumentParser(description="Train and evaluate the GloVe LSTM model")
parser.add_argument("--save", type=int, nargs="?", default=25, help="Save model every N epochs")
parser.add_argument("--epochs", type=int, nargs="?", default=None, help="Number of epochs")
args = parser.parse_args()

console.info("Will save every {} epochs for {} epochs".format(
    args.save,
    args.epochs if args.epochs is not None else "unlimited"))

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers import GloveClassifier

# Load data source
tweets_v3 = ["data/tweets.v3.part{:02d}.txt".format(n) for n in range(1, 11)]
data_src = TweetsDataSource(*tweets_v3, random_seed=5)

# Print info about the data distribution and MFC accuracy
mfc_class = np.argmax(np.bincount(data_src.train_labels))
console.info("train distribution", np.bincount(data_src.train_labels))
mfc_acc_train = np.equal(mfc_class, data_src.train_labels).mean()
mfc_acc_test  = np.equal(mfc_class, data_src.test_labels).mean()
console.info("train mfc", mfc_acc_train)
console.info("test mfc", mfc_acc_test)

lstm = GloveClassifier("/cse/web/homes/brandv2/nlp/glove.dict.200d.pkl")
lstm.train(
    data_src.train_inputs,
    data_src.train_labels,
    save_every_n_epochs=args.save,
    num_epochs=args.epochs,
    eval_tokens=data_src.test_inputs,
    eval_labels=data_src.test_labels,
    continue_previous=True)
