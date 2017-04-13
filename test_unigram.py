"""
Run UnigramClassifier on fake data.
"""
from classifiers import UnigramClassifier
from utility import console

class FakeDataSource():
    NUM_LABELS = 4

    def __init__(self):
        pass

    def get_inputs(self):
        return [
            "I love eggs for breakfast",
            "Flies love fruit",
            "I love bacon",
            "I love coffee",
            "I love chocolate",
            "I love , love chocolate and coffee"
            "I like spam sandwiches",
            "I like peas and onions",
            "I like water",
            "I like ice cream , especially chocolate",
            "I hate strawberry milk",
            "I hate swiss cheese",
            "I hate vegetables , especially asparagus",
            "I hate fruit",
            "I am Python",
            "I have no brain",
            "Chickens lay eggs",
            "Sam eats spam",
            "I drink water all day",
            "I drink coffee in the morning"
        ]

    def get_labels(self):
        return [
            3 if sent.find("love") != -1 else
            2 if sent.find("like") != -1 else
            1 if sent.find("hate") != -1 else
            0
            for sent in self.get_inputs()
        ]

    def decode_labels(self, labels, meanings=("neutral", "hatred", "enjoyment", "endearment")):
        return [meanings[i] for i in labels]

if __name__ == "__main__":
    classifier = UnigramClassifier()
    data_source = FakeDataSource()

    console.time("training")
    classifier.train(data_source.get_inputs(),
                     data_source.get_labels(),
                     data_source.NUM_LABELS)
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