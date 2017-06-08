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
from classifiers import LstmClassifier

data_src = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5, tokenizer="ours")

lstm = LstmClassifier()

lstm.train(
    data_src,
    save_every_n_epochs=args.save,
    num_epochs=args.epochs,
    save_hook=None,
    continue_previous=True)
