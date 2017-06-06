from data_source.tokenize import tokenize_tweet, wrapper
from data_source import TweetsDataSource
from collections import Counter
from utility import console

def analyze():

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
    console.log("Word_tokenize:")
    info(ds, ds.tokenize)
    console.log("Custom tokenizer:")
    info(ds, tokenize_tweet)


def summary():
    ds = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5, tokenizer="word")

    console.log("Writing summary...")
    tweets = ds.train_raw_inputs
    console.time("tokenize all tweets")
    tokenized = [tokenize_tweet(t) for t in tweets]
    console.time_end("tokenize all tweets")

    vocab = Counter(word for sent in tokenized for word in sent)
    with open("custom.out", "w") as f:
        f.write("# TWEET TOKENIZER TEST\n")
        f.write("# VOCAB SIZE: {}\n".format(len(vocab)))
        f.write("# WORDS WITH ONE OCCURRENCE: {}\n".format(sum(count == 1 for count in vocab.values())))
        for i in range(len(tweets)):
            f.write("[{}/{}]\n".format(i+1, len(tweets)))
            f.write("ORIG: " + tweets[i] + "\n")
            f.write("TOKS: " + " ".join(tokenized[i]) + "\n\n")
    return vocab



def test():
    cases = [
        (
            "Elongations",
            "hahahahha lololol hahahah wtffffffff omgggg",
            "haha lol haha wtfff <elong> omggg <elong>".split()
        ),
        (
            "Contractions and apostrophes",
            "don't should've i'm bob's 'quote' you're we'll",
            "do n't should 've i 'm bob 's ' quote ' you 're we 'll".split()
        ),
        (
            "Punctuation",
            "ya? ya. ya: ya! ya, \"ya\"",
            "ya ? ya . ya : ya ! ya , \" ya \"".split()
        ),
        (
            "User, allcaps, and apostrophes interact ok",
            "@BOB's BOB's",
            "<user> 's bob <allcaps> 's".split()
        ),
        (
            "Complex",
            "OMGGGGGGGG, tokenize those #HashTags, lolol "
                + "@mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) hahahahahaha!!!!!",
            ("omggg <elong> <allcaps> , tokenize those <hashtag> hashtags , lol "
                + "<user> and <number> ( <url> ) . w / <heart> <smile> haha ! <repeat>").split()
        ),
    ]

    def run_case(case):
        desc, tweet, want = case
        got = tokenize_tweet(tweet)
        if got != want:
            console.warn("FAIL:", desc)
            console.warn("Got", got)
            console.warn("Want", want)
            return False
        else:
            console.log(console.colors.GREEN + "OK: ", desc, console.colors.END)
            return True


    def compare(case):
        desc, tweet, want = case
        desc += " are the same?"
        v1 = tokenize_tweet(tweet)
        v2, _ = wrapper(tweet)
        if v1 != v2:
            console.warn("FAIL:", desc)
            console.warn("Got", v2)
            console.warn("Want", v1)
            return False
        else:
            console.log(console.colors.GREEN + "OK: ", desc, console.colors.END)
            return True
    
    for case in cases:
        run_case(case)
    for case in cases:
        compare(case)

test()
