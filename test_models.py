"""
Try running the models
"""
from classifiers import UnigramClassifier, LstmClassifier
from data_source import TweetsDataSource
from utility import console, Emotion
import numpy as np
import random

tweets_data_src = TweetsDataSource("data/tweets.v2.txt",
                                   "data/tweets.v2.part2.txt",
                                   "data/tweets.v2.part3.txt",
								   random_seed=5)

def print_metrics(title, truth, predictions, mfc_class):
    N = len(truth)
    frequencies = np.bincount(truth) / N
    mfc_acc = frequencies[mfc_class]
    acc = np.equal(predictions, truth).mean()

    console.h1(title)
    console.h1("\tFrequencies: {}".format(frequencies))
    console.h1("\tCount:\t{}".format(N))
    console.h1("\tAcc:\t{:.4f}".format(acc))
    console.h1("\tMfc:\t{:.4f}".format(mfc_acc))

    if mfc_acc >= acc:
        console.warn("Your classifier isn't better than the MFC!")

    for i in range(len(Emotion)):
        console.h1("Label {} ({})".format(i, Emotion(i).name))

        truth_is_pos = np.equal(truth, i)
        pred_is_pos  = np.equal(predictions, i)

        if truth_is_pos.sum() == 0 and pred_is_pos.sum() == 0:
            console.h1("\t[absent]")
            continue

        true_pos     = np.logical_and(truth_is_pos, pred_is_pos).sum()
        false_pos    = np.logical_and(~truth_is_pos, pred_is_pos).sum()
        true_neg     = np.logical_and(~truth_is_pos, ~pred_is_pos).sum()
        false_neg    = np.logical_and(truth_is_pos, ~pred_is_pos).sum()

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * (precision * recall) / (precision + recall)


        console.h1("\tPrec:\t{:.4f}".format(precision))
        console.h1("\tRec:\t{:.4f}".format(recall))
        console.h1("\tF1:\t{:.4f}".format(f1))



def assess_classifier(classifier, data_src):
    mfc_class = np.bincount(data_src.train_labels).argmax()
    print_metrics(
        "Training data",
        data_src.train_labels,
        classifier.predict(data_src.train_inputs),
        mfc_class)
    print_metrics(
        "Test data",
        data_src.test_labels,
        classifier.predict(data_src.test_inputs),
        mfc_class)


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


if __name__ == "__main__":
    for label in tweets_data_src.test_labels:
        assert Emotion(label).value == label
    for label in tweets_data_src.train_labels:
        assert Emotion(label).value == label

    console.h1("UnigramClassifier")
    unigram = UnigramClassifier(len(Emotion))
    unigram.train(tweets_data_src.train_inputs, tweets_data_src.train_labels, max_epochs=10000)
    assess_classifier(unigram, tweets_data_src)
    
    console.h1("-" * 80)
    console.h1("LstmClassifier")
    lstm_input = RawInputsWrapper(tweets_data_src)
    assess_classifier(LstmClassifier(), lstm_input)


