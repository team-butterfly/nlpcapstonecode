"""
Tune parameters of UnigramClassifier
"""
from classifiers import UnigramClassifier, LstmClassifier
from data_source import FakeDataSource, TweetsDataSource
from utility import console
import numpy as np

tweets_data_src = TweetsDataSource("data/tweets.v2.txt", random_seed=5)

def assess_classifier(classifier, data_src, max_epochs):
    """Run the classifier, print metrics, and return accuracy on dev set"""
    console.time("training")
    classifier.train(
        data_src.train_inputs,
        data_src.train_labels,
        max_epochs=max_epochs)
    console.time_end("training")

    console.time("predicting")
    train_predictions = classifier.predict(data_src.train_inputs)
    test_predictions = classifier.predict(data_src.test_inputs)
    console.time_end("predicting")

    n_train = len(data_src.train_inputs)
    train_frequencies = np.bincount(data_src.train_labels) / n_train
    train_mfc = train_frequencies.max()
    train_acc = np.equal(train_predictions, data_src.train_labels).mean()

    n_test = len(data_src.test_inputs)
    test_frequencies = np.bincount(data_src.test_labels) / n_test
    test_mfc = test_frequencies[train_frequencies.argmax()]
    test_acc = np.equal(test_predictions, data_src.test_labels).mean()

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
        console.warn("Your classifier isn't better than the MFC!")

    return test_acc

def find_good_unk_threshold():
    """Best unk_thresholds were 7, 4, 3?"""
    accs = {}
    for unk in range(1, 50):
        console.h1("Trying unk_threshold={}".format(unk))
        clsf = UnigramClassifier(data_src.num_labels, unk_threshold=unk)
        test_acc = assess_classifier(clsf, tweets_data_src, 10000)
        accs[unk] = test_acc
    sorted_accs = sorted(accs.items(), key=lambda it: it[1], reverse=True) # Sort the dict by ascending values
    print(sorted_accs)


def run_lstm():
    console.h1("Running LstmClassifier")

    class RawInputsWrapper():
        def __init__(self, data_src):
            self._data_src = data_src
            
        @property
        def train_inputs(self):
            return self._data_src.train_raw_inputs

        @property
        def train_labels(self):
            return self._data_src.train_labels

        @property
        def test_inputs(self):
            return self._data_src.test_raw_inputs

        @property
        def test_labels(self):
            return self._data_src.test_labels

    clsf = LstmClassifier(5)
    assess_classifier(clsf, RawInputsWrapper(tweets_data_src), 10)

if __name__ == "__main__":
    run_lstm()