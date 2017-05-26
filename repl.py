"""Read-eval-predict loop for a classifier"""
from sys import argv, exit

if len(argv) != 2:
    print("Supply model name")
    exit(1)

import numpy as np
from data_source import TweetsDataSource
from classifiers import GloveClassifier
from utility import Emotion

g = GloveClassifier(argv[1])
ds = TweetsDataSource()

while True:
    sent = input("Enter sentence: ")
    if sent.rstrip() == "":
        continue
    sent = ds.tokenize(sent)
    tokens, soft_labels, attns = g.predict_soft_with_attention([sent])[0]
    print("tokens:", tokens)
    print("attention:", attns.tolist())
    print("soft_labels:", soft_labels)
