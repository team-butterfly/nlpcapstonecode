"""
Reads a text file of word embeddings into a pickled dictionary.
Loading them from a pickle is about 60x faster than re-parsing the text every
time, and takes up less disk space.
"""
import numpy as np
import pickle
import sys

def read_glove(path_in):
    with open(path_in) as f:
        print("Reading from", path_in)
        tokens = f.readlines()
        print("Done reading")

    vocab_size = len(tokens)
    dim = len(tokens[0].rstrip().split(" "))-1
    print("Embedding dimension looks like", dim)
    
    embeddings = np.empty([vocab_size, dim], np.float32)
    index_word = np.empty(vocab_size, dtype="<U64")
    word_index = {}
    for i, toks in enumerate(tokens):
        line = toks.rstrip().split(" ")
        word = line[0]
        embeddings[i] = line[1:]
        index_word[i] = word
        word_index[word] = i
    print("Vocab size: {} words".format(vocab_size))
    
    return {
        "vocab_size": vocab_size,
        "dimension": dim,
        "embeddings": embeddings,
        "index_word": index_word,
        "word_index": word_index
    }

def compress_glove(path_in, path_save):
    glove_dict = read_glove(path_in)
    print("Pickling to", path_save)
    with open(path_save, "wb") as f:
        pickle.dump(glove1, f)
    print("Done")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Args: [1] embeddings file name (e.g. 'glove.twitter.27B.200d.txt')")
        print("      [2] pickle file name (e.g. 'glove.dict.200d.pkl')")
        sys.exit(1)
    compress_glove(sys.argv[1], sys.argv[2])
