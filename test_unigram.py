"""
Run UnigramClassifier on fake data.
"""
from classifiers import UnigramClassifier
from utility import console
from data_source import FakeDataSource
import nltk

if __name__ == "__main__":
    classifier = UnigramClassifier()
    data_source = FakeDataSource()

    console.time("training")
    classifier.train(data_source.train_inputs,
                     data_source.train_labels,
                     data_source.num_labels)
    console.time_end("training")

    console.time("predicting")
    predictions = classifier.predict(data_source.test_inputs)
    console.time_end("predicting")

    console.h1("UnigramClassifier predictions:")
    for sentence, label in zip(data_source.test_inputs, data_source.decode_labels(predictions)):
        console.h1(" ".join(sentence))
        console.h1("\t->", label)
