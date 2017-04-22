"""
Run UnigramClassifier on fake data.
"""
from classifiers import UnigramClassifier
from utility import console
from data_source import FakeDataSource, TweetsDataSource
import numpy as np

data_source = TweetsDataSource("data/tweets.v2.txt", random_seed=5)

for unk in range(2, 50):
    classifier = UnigramClassifier(data_source.num_labels, unk_threshold=unk)

    console.time("training")
    classifier.train(data_source.train_inputs, data_source.train_labels, max_epochs=10000)
    console.time_end("training")

    console.time("predicting")
    train_predictions = classifier.predict(data_source.train_inputs)
    test_predictions = classifier.predict(data_source.test_inputs)
    console.time_end("predicting")

    n_train = len(data_source.train_inputs)
    train_frequencies = np.bincount(data_source.train_labels) / n_train
    train_mfc = train_frequencies.max()
    train_acc = np.equal(train_predictions, data_source.train_labels).mean()

    n_test = len(data_source.test_inputs)
    test_frequencies = np.bincount(data_source.test_labels) / n_test
    test_mfc = test_frequencies.max()
    test_acc = np.equal(test_predictions, data_source.test_labels).mean()

    console.h1("Training data:")
    console.h1("\tTrain frequencies {}".format(train_frequencies))
    console.h1("\tCount:\t{}".format(n_train))
    console.h1("\tAcc:\t{}".format(train_acc))
    console.h1("\tMfc:\t{}".format(train_mfc))

    console.h1("Testing data:")
    console.h1("\tTest frequencies {}".format(test_frequencies))
    console.h1("\tCount:\t{}".format(n_test))
    console.h1("\tAcc:\t{}".format(test_acc))
    console.h1("\tMfc:\t{}".format(test_mfc))

    if test_mfc >= test_acc:
        console.warn("Your model isn't better than the MFC!")

    """
    for tweet_tokens, label in zip(data_source.test_inputs, data_source.decode_labels(test_predictions)):
        console.log(" ".join(tweet_tokens))
        console.log(label)
    """