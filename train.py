"""
Runs a simple train/eval loop of the GloVe LSTM
"""
import argparse
from utility import console

parser = argparse.ArgumentParser(description="Train and evaluate the GloVe LSTM model")
parser.add_argument("--glove", action="store_true", help="Use GloVe Twitter vocabulary")
parser.add_argument("--name", type=str, required=True, help="Name of this training instance. "
	+ " This determines where to save Tensorboard logs and ckpts.")
parser.add_argument("--save_interval", type=int, nargs="?", default=None, help="Save model every N epochs")
parser.add_argument("--eval_interval", type=int, nargs="?", default=None, help="Evaluate model every N epochs")
parser.add_argument("--progress_interval", type=float, nargs="?", default=0.01, help="Progress bar interval (percent)")
parser.add_argument("--num_epochs", type=int, nargs="?", default=None, help="Number of epochs")
parser.add_argument("--tokenizer", type=str, required=False, help="Tokenize function")
parser.add_argument("--logits_mode", type=str, required=False, default="attn", help="Logits mode")
args = parser.parse_args()

if args.save_interval is not None:
	console.info(
		"Will save every {} epochs for {} epochs".format(
		args.save_interval,
		args.num_epochs if args.num_epochs is not None else "unlimited"))

import numpy as np
from utility import console
from data_source import TweetsDataSource
from classifiers.glove import GloveTraining
from classifiers.customvocab import CustomVocabTraining
from classifiers.utility import HParams

console.info("Run name:", args.name)
hparams = HParams() # Hyperparameters "struct"
hparams.batch_size = 200 
hparams.vocab_size = 50000 
console.info("HParams:\n", hparams)

if args.glove:
    console.info("Using GloVe vocabulary...")
    training = GloveTraining(args.name, hparams)
else:
    console.info("Using custom vocabulary...")
    training = CustomVocabTraining(args.name, hparams, logits_mode=args.logits_mode)

# Load data source
data_src = TweetsDataSource(file_glob="data/tweets.v3.part*.txt",
	random_seed=5) # , tokenizer=args.tokenizer if args.tokenizer is not None else 'tweet')

# Print info about the data distribution and MFC accuracy
mfc_class = np.argmax(np.bincount(data_src.train_labels))
console.info("Train distribution:", np.bincount(data_src.train_labels)[1:])
mfc_acc_train = np.equal(mfc_class, data_src.train_labels).mean()
mfc_acc_test  = np.equal(mfc_class, data_src.test_labels).mean()
console.info("Train mfc:", mfc_acc_train)
console.info("Test mfc:", mfc_acc_test)

run_name = args.name
del args.name
del args.glove
del args.tokenizer
del args.logits_mode

def log_hparams():
    with open("log/runs.txt", "a") as logf:
        logf.write(run_name + ":\n")
        logf.write(str(hparams) + "\n")

try:
    training.run(data_src, **vars(args))
    log_hparams()
except (Exception, KeyboardInterrupt) as e:
    console.warn(e, e.args)
    console.log("\nTraining interrupted. Save this session (y/n)? ")
    if input().startswith("y"):
        log_hparams()
    else:
        training.erase_files()
