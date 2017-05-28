"""
Modified version of script by Motoki Wu
"""

import sys
import re

FLAGS = re.MULTILINE | re.DOTALL
CONTRACTIONS = ["n't", "'d", "'ll", "'s", "'ve", "'m", "'re"]
PUNCTUATION = r"!?.,()\[\]\-:;\"~"

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    result = "<hashtag> " + hashtag_body
    # if hashtag_body.isupper():
    #     result += " <allcaps>"
    return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> "


def tokenize_tweet(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"/"," / ")
    text = re_sub(r"@\w+", " <user> ")
    text = re_sub(r"\b{}{}[)dD]+|[)dD]+{}{}\b".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"\b{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"\b{}{}\(+|\)+{}{}\b".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"\b{}{}[\/|l*]\b".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3"," <heart> ")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.])\1+", r"\1 <repeat> ")


    text = re_sub(r"\b([A-Z]){2,}\b", allcaps)

    text = text.lower()

    # If a word ends in >3 of same character, truncate to 3
    # and add <elong> token
    text = re_sub(r"\b(\S*?)(.)\2\2{2,}\b", r"\1\2\2\2 <elong> ")

    # Expand contractions
    for contr in CONTRACTIONS:
        text = re_sub(r"({})\b".format(contr), r" \1")

    # Expand punctuation
    text = re_sub(r"([{}]+)".format(PUNCTUATION), r" \1 ")

    # Contract "hahaha..." into "haha"
    text = re_sub(r"\b(haha)(?:ha)+h?\b", r"\1 <elong> ")
    
    # Contract "lolol..." into "lol"
    text = re_sub(r"\b(lol)(?:ol)\b", r"\1 <elong> ")

    return text.split()


def example(text):
    if text == "TEST":
        text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
    print(tokenize_tweet(text))


def analyze():
    from collections import Counter
    from .data_source import TweetsDataSource
    from utility import console
    
    def info(ds, tok_func):
        tweets = ds.train_raw_inputs
        
        console.log("\tTokenizing all {} tweets...".format(len(tweets)))
        console.time("tokenize")
        tokenized = [tok_func(tweet.lower()) for tweet in tweets]
        console.time_end("tokenize")

        count = Counter(word for sent in tokenized for word in sent)
        once = sum(n == 1 for n in count.values())
        console.log("\tVocab size: {}".format(len(count)))
        console.log("\tLength of corpus: {}".format(sum(count.values())))
        console.log("\t{}/{} tokens appear only once ({:.3f}%)".format(once, len(count), once / len(count) * 100))
        console.log("\tAverage word frequency: {:.2f}".format(sum(count.values()) / len(count)))

        n = 100
        console.log("{} most frequent:".format(n))
        console.log(count.most_common(n))

    ds = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5, tokenizer="word")
    """
    console.log("Word_tokenize:")
    info(ds, ds.tokenize)
    console.log("Custom tokenizer:")
    info(ds, tokenize_tweet)
    """

    console.log("saving custom results...")
    tweets = ds.train_raw_inputs
    with open("custom.out", "w") as f:
        for tweet in tweets:
            toks = tokenize_tweet(tweet)
            f.write("ORIG: " + tweet + "\n")
            f.write("TOKS: " + " ".join(toks) + "\n\n")
