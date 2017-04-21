# main file that runs all the things
import argparse
import os
from utility import console, Emotion
from classifiers import UnigramClassifier
from tts import TTS
from data_source import FakeDataSource, TweetsDataSource
import numpy as np

def text_to_speech(classifier, data_source, text, output_path):

    tts = TTS()
    emotion = classifier.predict([text])[0]

    return tts.speak(text, Emotion[data_source.decode_labels([emotion])[0]], output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-aware text-to-speech")
    parser.add_argument("text", nargs="*", default=[])

    args = parser.parse_args()


    if len(args.text) == 0:
        text = "Emotion aware text to speech is really great! Thank you for trying it out."
    else:
        text = " ".join(args.text)

    data_source = TweetsDataSource("data/tweets-small.txt")
    classifier = UnigramClassifier(data_source.num_labels, 2)

    console.time("training")
    classifier.train(data_source.train_inputs, data_source.train_labels)
    console.time_end("training")

    console.time("predicting")
    train_predictions = classifier.predict(data_source.train_inputs)
    test_predictions = classifier.predict(data_source.test_inputs)
    console.time_end("predicting")

    n_train = len(data_source.train_inputs)
    train_frequencies = np.bincount(data_source.train_labels) / n_train
    train_mfc = 1-train_frequencies.max()
    train_acc = np.sum(np.equal(train_predictions, data_source.train_labels)) / n_train

    n_test = len(data_source.test_inputs)
    test_frequencies = np.bincount(data_source.test_labels) / n_test
    test_mfc = 1-test_frequencies.max()
    test_acc = np.sum(np.equal(test_predictions, data_source.test_labels)) / n_test

    console.h1("Training data:")
    console.h1("\tCount:\t{}".format(n_train))
    console.h1("\tAcc:\t{}".format(train_acc))
    console.h1("\tMfc:\t{}".format(train_mfc))

    console.h1("Testing data:")
    console.h1("\tCount:\t{}".format(n_test))
    console.h1("\tAcc:\t{}".format(test_acc))
    console.h1("\tMfc:\t{}".format(test_mfc))

    output_path = os.path.join("/tmp", "_".join(text.split()) + ".aif")
    output_path = text_to_speech(classifier, data_source, text, output_path)
    console.log("Writing to", output_path)
    os.system("afplay '{}' || ffplay '{}' || play '{}'".format(output_path, output_path, output_path))
