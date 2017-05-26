"""
Prints performance metrics of various models.
"""
from classifiers import UnigramClassifier, LstmClassifier, EmoLexBowClassifier, GloveClassifier
from data_source import TweetsDataSource
from utility import console, Emotion
import numpy as np

EPS = 1E-8

def print_metrics(truth, predictions, mfc_class):
    N = len(truth)
    frequencies = np.bincount(truth) / N
    mfc_acc = frequencies[mfc_class]
    acc = np.equal(predictions, truth).mean()

    console.h1("\tFrequencies: {}".format(frequencies[1:]))
    console.h1("\tCount:\t{}".format(N))
    console.h1("\tAcc:\t{:.4f}".format(acc))
    console.h1("\tMfc:\t{:.4f}".format(mfc_acc))

    if mfc_acc >= acc:
        console.warn("Your classifier isn't better than the MFC!")

    console.h1("\t" + "-"*30)

    for i in range(len(Emotion)):
        console.h1("\tLabel {} ({})".format(i, Emotion(i).name))

        truth_is_pos = np.equal(truth, i)
        pred_is_pos  = np.equal(predictions, i)

        if truth_is_pos.sum() == 0 and pred_is_pos.sum() == 0:
            console.h1("\t\t[absent]")
            continue

        true_pos  = np.logical_and(truth_is_pos, pred_is_pos).sum()
        false_pos = np.logical_and(~truth_is_pos, pred_is_pos).sum()
        true_neg  = np.logical_and(~truth_is_pos, ~pred_is_pos).sum()
        false_neg = np.logical_and(truth_is_pos, ~pred_is_pos).sum()

        precision = true_pos / (true_pos + false_pos + EPS)
        recall = true_pos / (true_pos + false_neg + EPS)
        f1 = 2 * (precision * recall) / (precision + recall + EPS)

        console.h1("\t\tPrec:\t{:.4f}".format(precision))
        console.h1("\t\tRec:\t{:.4f}".format(recall))
        console.h1("\t\tF1:\t{:.4f}".format(f1))


def assess_classifier(classifier, data_src):
    mfc_class = np.bincount(data_src.train_labels).argmax()
    print_metrics(
        data_src.test_labels,
        classifier.predict(data_src.test_inputs),
        mfc_class)


class RawInputsWrapper():
    def __init__(self, data_src):
        self.train_inputs = data_src.train_raw_inputs
        self.train_labels = data_src.train_labels
        self.test_inputs = data_src.test_raw_inputs
        self.test_labels = data_src.test_labels


if __name__ == "__main__":
    tweets = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5, tokenizer="word")

    console.h1("UnigramClassifier")
    unigram = UnigramClassifier()
    unigram.train(tweets.train_inputs, tweets.train_labels, max_epochs=1000)
    assess_classifier(unigram, tweets)
    del unigram
    
    console.h1("EmoLexBowClassifier")
    emolex = EmoLexBowClassifier()
    emolex.train(tweets.train_inputs, tweets.train_labels)
    assess_classifier(emolex, tweets)
    del emolex

    console.h1("GloveClassifier - Attention on Hidden States")
    glove = GloveClassifier("attention_hiddens")
    assess_classifier(glove, tweets)
    glove.close()
    del glove

    console.h1("GloveClassifier - No Attention")
    glove = GloveClassifier("no_attention")
    assess_classifier(glove, tweets)
    glove.close()
