# main file that runs all the things
import argparse
import os
from utility import console, Emotion
from classifiers import UnigramClassifier
from tts import TTS
from data_source import FakeDataSource, TweetsDataSource
import numpy as np

def text_to_speech(classifier, data_source, text):
    tts = TTS()
    output_path = "/tmp/" + tts.as_file_path(text) + ".aif"
    emotion = Emotion[data_source.decode_labels([classifier.predict([text])[0]])[0]]
    console.log(emotion)
    return tts.speak(text, emotion, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-aware text-to-speech")
    parser.add_argument("text", nargs="*", default=[])

    args = parser.parse_args()


    if len(args.text) == 0:
        text = "Emotion aware text to speech is really great! Thank you for trying it out."
    else:
        text = " ".join(args.text)

    data_source = TweetsDataSource("data/tweets.v2.txt", random_seed=5)
    classifier = UnigramClassifier(data_source.num_labels, unk_threshold=9)

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
    console.h1("\tCount:\t{}".format(n_train))
    console.h1("\tAcc:\t{}".format(train_acc))
    console.h1("\tMfc:\t{}".format(train_mfc))

    console.h1("Testing data:")
    console.h1("\tCount:\t{}".format(n_test))
    console.h1("\tAcc:\t{}".format(test_acc))
    console.h1("\tMfc:\t{}".format(test_mfc))

    while not text.startswith("!"):
        text = input("Enter your text (! to quit): ")
        output_path = text_to_speech(classifier, data_source, text)
        # console.log("Writing to", output_path)
        os.system("afplay '{}' || ffplay '{}' || play '{}'".format(output_path, output_path, output_path))
