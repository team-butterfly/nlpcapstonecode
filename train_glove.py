"""
Runs a simple train/eval loop of the GloVe LSTM
"""
import argparse
from utility import console

parser = argparse.ArgumentParser(description="Train and evaluate the GloVe LSTM model")
parser.add_argument("--save", type=int, nargs="?", default=1, help="Save model every N epochs")
parser.add_argument("--epochs", type=int, nargs="?", default=None, help="Number of epochs")
parser.add_argument("--name", type=str, required=True, help="Name of this run. This decides where to save Tensorboard logs and ckpts.")
args = parser.parse_args()

console.info("Run name:", args.name)
console.info(
    "Will save every {} epochs for {} epochs".format(
    args.save,
    args.epochs if args.epochs is not None else "unlimited"))

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers.glove import GloveTraining, HParams

hparams = HParams()
hparams.batch_size = 200
console.info("HParams:\n", hparams)
g = GloveTraining(args.name, hparams)

# Load data source
data_src = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5)

# Print info about the data distribution and MFC accuracy
mfc_class = np.argmax(np.bincount(data_src.train_labels))
console.info("Train distribution:", np.bincount(data_src.train_labels))
mfc_acc_train = np.equal(mfc_class, data_src.train_labels).mean()
mfc_acc_test  = np.equal(mfc_class, data_src.test_labels).mean()
console.info("Train mfc:", mfc_acc_train)
console.info("Test mfc:", mfc_acc_test)

g.run(
    data_src,
    save_every_n_epochs=args.save,
    num_epochs=args.epochs)
