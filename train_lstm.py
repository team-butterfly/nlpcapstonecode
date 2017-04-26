import argparse
from utility import console

parser = argparse.ArgumentParser(description="Run the LSTM model")
parser.add_argument("--save", type=int, nargs="?", default=25, help="Save model every N epochs")
args = parser.parse_args()

console.info("Will save every {} epochs".format(args.save))

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers import LstmClassifier

data_src = TweetsDataSource("data/tweets.v2.txt", "data/tweets.v2.part2.txt", random_seed=5)

lstm = LstmClassifier()
lstm.train(
    data_src.train_raw_inputs,
    data_src.train_labels,
    save_every_n_epochs=args.save,
    continue_previous=True)
