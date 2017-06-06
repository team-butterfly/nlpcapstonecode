"""Read-eval-predict loop for a classifier"""
from sys import argv, exit

if len(argv) != 2:
    print("Supply model name")
    exit(1)

import numpy as np
from data_source import TweetsDataSource
from classifiers import CustomVocabClassifier
from utility import Emotion

g = CustomVocabClassifier(argv[1])
ds = TweetsDataSource(tokenizer="ours")

while True:
    sent = input("Enter sentence: ")
    # Empty string causes TF to crash
    if sent.rstrip() == "":
        continue
    
    tokens, soft_labels, attns = g.predict_soft_with_attention([sent])[0]

    print("tokens is", tokens)
    print("softlabels is", soft_labels)
    print("attens is", attns)
    
    # Print tokens with attention
    print("Attention:")
    maxlen = max(len(t) for t in tokens)
    form = "    {{:{}s}} {{:.2f}}%".format(maxlen)
    for tok, attn in zip(sent.split(), attns):
        print(form.format(tok, attn * 100))

    print("Predictions:")
    # Print emotions in descending order
    keys = sorted(soft_labels.items(), key=lambda kv: kv[1], reverse=True)
    for em, frac in keys:
        print("    {:8s} {:.2f}%".format(em.name, frac * 100))
    print()
