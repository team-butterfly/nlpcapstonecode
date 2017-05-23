import pickle
import sys

def make_vocab(path_in, path_out):
    g = pickle.load(open(path_in, "rb"))
    with open(path_out, "w") as f:
        for word in g["index_word"]:
            f.write(word + "\n")
    with open(path_out, "r") as f:
        for i, want in enumerate(g["index_word"]):
            got = f.readline()[:-1]
            if got != want:
                print("{}: want '{}'; got '{}'".format(i, want, got))
        f.seek(0)
        want = len(g["index_word"])
        got = len(f.readlines())
        if got != want:
            print("want len='{}'; got len='{}'".format(i, want, got))
    print("Done")

make_vocab("glove.dict.200d.pkl", "vocab.glove.txt")
