import argparse

parser = argparse.ArgumentParser(description="Run the LSTM model")
parser.add_argument("--mode",
                    type=str,
                    choices=["train", "eval"],
                    help="Choose what operation to run")
args = parser.parse_args()

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers import LstmClassifier

data_src = TweetsDataSource("data/tweets.v2.txt", random_seed=5)
lstm = LstmClassifier()

if args.mode == "train":
    
    console.info("Training mode")
    lstm.train(
        data_src.train_raw_inputs,
        data_src.train_labels,
        save_every_n_epochs=10, 
        continue_previous=True)

elif args.mode == "eval":
    
    console.info("Evaluation mode")
    acc_test = np.equal(lstm.predict(data_src.test_raw_inputs), data_src.test_labels).mean()
    acc_train = np.equal(lstm.predict(data_src.train_raw_inputs), data_src.train_labels).mean()
    console.h1("\tTest accuracy:", acc_test)
    console.h1("\tTrain accuracy:", acc_train)
