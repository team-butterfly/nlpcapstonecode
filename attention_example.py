import nltk
from utility import Emotion
from classifiers import GloveClassifier

g = GloveClassifier("glove.dict.200d.pkl")
while True:
    sent = input("Enter sentence: ")
    y = g.predict_soft_with_attention([sent])[0]
    print(y)
