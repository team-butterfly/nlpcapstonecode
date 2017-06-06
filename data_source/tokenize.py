"""
Modified version of script by Motoki Wu
"""

import sys
import re

FLAGS = re.MULTILINE | re.DOTALL
APOSTROPHE_PAT = "['’]"
CONTRACTIONS = {"n_t", "_d", "_ll", "_s", "_ve", "_m", "_re"}
CONTRACTIONS = {r"(\w*)(" + pat.replace("_", APOSTROPHE_PAT) + ")" for pat in CONTRACTIONS}
PUNCTUATION = r"!?.,()\[\]\-:;\"~*\\/#@^&|{}+="


def allcaps(match):
    return match.group().lower() + " <allcaps>"

# Isolate apostrophes except when they are in contractions
def apostrophe(match):
    for contr in CONTRACTIONS:
        contr_match = re.match(contr, match.group(0))
        if contr_match is not None:
            return " ".join(contr_match.groups())
    return " ".join(match.group(1, 2, 3))



def tokenize_tweet(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
    text = re_sub(r"<3"," <heart>")
    text = re_sub(r"#(\S+)", r" <hashtag> \1")
    text = re_sub(r"([!?.])\1+", r"\1 <repeat> ")

    text = re_sub(r"@\w+", " <user> ")
    text = re_sub(r"\b([A-Z]){2,}\b", allcaps)

    text = text.lower()

    # If a word ends in >3 of same character, truncate to 3 and add <elong> token
    text = re_sub(r"\b(\S*?)(.)\2\2{2,}\b", r"\1\2\2\2 <elong> ")

    text = re_sub(r"(\w*)({})(\w*)".format(APOSTROPHE_PAT), apostrophe)
    text = re_sub(r"\b[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")

    # Expand punctuation
    text = re_sub(r"([{}])\1*".format(PUNCTUATION), r" \1 ")

    # Contract "hahaha..."
    text = re_sub(r"\bha+(h+a+)+h*\b", r"haha")
    
    # Contract "lolol..."
    text = re_sub(r"\bl+o+l+(?:o+l+)*\b", r"lol")

    return text.split()

def wrapper(tweet):
    words = tweet.split()
    tokens = []
    mapping = []
    for i, word in enumerate(words):
        for token in tokenize_tweet(word):
            tokens.append(token)
            mapping.append(i)
    return tokens, mapping
