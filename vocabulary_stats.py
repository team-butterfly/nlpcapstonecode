"""
Prints stats about the GloVe embeddings and vocabulary of our Twitter data
"""

import pickle
from collections import Counter
from data_source import TweetsDataSource

table = []

glove = pickle.load(open("glove.dict.200d.pkl", "rb"))
table.append( ("GloVe vocabulary size", len(glove["index_word"])) )
table.append( ("GloVe embedding dimension", 200) )

ds = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5, tokenizer="ours")
c_raw = Counter(word.lower() for sent in ds.train_raw_inputs for word in sent.split(" "))
c_tokenized = Counter(word for sent in ds.train_inputs for word in sent)

table.append( ("Num. training tweets", len(ds.train_inputs)) )
table.append( ("Num. validation tweets", len(ds.test_inputs)) )

for name, counts in [("split on spaces", c_raw), ("smart tokenizer", c_tokenized)]:
    table.append( ("Num. unique tokens, " + name, len(counts)) )
    table.append( ("Num. tokens that only appear once", sum(c == 1 for c in counts.values())) )
    table.append( ("Percent of above", sum(c == 1 for c in counts.values()) / len(counts)) )
    table.append( ("Percent of tokens in GloVe", sum(c in glove["word_index"] for c in counts.keys()) / len(counts)) )

longest = max(len(name) for name, value in table)
row_fmt = "{:%ds} & {}" % longest

print(r"\begin{tabular}{l|l}")
for i, (name, val) in enumerate(table):
    end = " \\\\\n" if i != len(table)-1 else "\n"
    print("    " + row_fmt.format(name, val), end=end)
print(r"\end{tabular}")
