"""
Run UnigramClassifier on fake data.
"""
from classifiers import UnigramClassifier
from utility import console
from data_source import FakeDataSource

if __name__ == "__main__":
    classifier = UnigramClassifier()
    data_source = FakeDataSource()

    console.time("training")
    classifier.train(data_source.get_inputs(),
                     data_source.get_labels(),
                     data_source.num_labels())
    console.time_end("training")

    test_data = [
        "Anyone with a brain knows that dogs like coffee",
        "I ate spam for dinner",
        "We all love NLP",
        "Cats really love milk",
        "Today is Wednesday",
        "Everyone knows that snakes love eggs",
        "Cats absolutely hate the sensation of water on their skin",
    ]

    console.time("predicting")
    predictions = classifier.predict(test_data)
    console.time_end("predicting")

    console.h1("UnigramClassifier predictions:")
    for x, y in zip(test_data, data_source.decode_labels(predictions)):
        console.h1('"' + x + '"')
        console.h1("\t->", y)