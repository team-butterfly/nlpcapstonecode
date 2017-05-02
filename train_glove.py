import argparse
from utility import console

parser = argparse.ArgumentParser(description="Run the LSTM model")
parser.add_argument("--save", type=int, nargs="?", default=25, help="Save model every N epochs")
parser.add_argument("--epochs", type=int, nargs="?", default=None, help="Number of epochs")
args = parser.parse_args()

console.info("Will save every {} epochs".format(args.save))

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers import GloveClassifier

data_src = TweetsDataSource(
    "data/tweets.v2.txt",
    "data/tweets.v2.part2.txt",
    "data/tweets.v2.part3.txt",
    random_seed=5)

lstm = GloveClassifier("glove.dict.200d.pkl")
lstm.train(
    data_src.train_inputs,
    data_src.train_labels,
    save_every_n_epochs=args.save,
    num_epochs=args.epochs,
    eval_tokens=data_src.test_inputs,
    eval_labels=data_src.test_labels,
    continue_previous=True)
