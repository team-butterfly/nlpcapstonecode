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

data_src = TweetsDataSource(
    "data/tweets.v2.txt",
    "data/tweets.v2.part2.txt",
    "data/tweets.v2.part3.txt",
    random_seed=5)

lstm = LstmClassifier()

best = 0
best_step = 0
def evaluate(epoch, step):
    global best, best_step
    acc = np.equal(
        lstm.predict(data_src.test_raw_inputs),
        data_src.test_labels).mean()
    if acc > best:
        best = acc
        best_step = step
    console.log("Test accuracy: {:.5f} (best {:.5f} at step {})".format(acc, best, best_step))

try:
    lstm.train(
        data_src.train_raw_inputs,
        data_src.train_labels,
        save_every_n_epochs=args.save,
        num_epochs=args.epochs,
        save_hook=evaluate,
        continue_previous=True)
except Exception as e:
    console.warn(e)
finally:
    console.log("Terminated; best test accuracy was {:.5f} at step {}".format(best, best_step))
