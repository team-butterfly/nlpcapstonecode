from utility import console, Emotion
from data_source import DataSource

class Classifier:
    def train(self, data):
        return None

    # Returns one of the Emotion constants
    def predict(self, text):
        return None


class UnigramClassifier(Classifier):
    def __init__(self):
        return None

if __name__ == "__main__":
    # run training code
    classifier = UnigramClassifier()
    data = DataSource()
    classifier.train(data)
