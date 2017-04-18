#!/usr/bin/env python2

# TODO: Port to Python 3 and fix Unicode handling

import codecs
import emoji
import itertools
import tweepy
import re

from utility.emotion import Emotion

# TODO: handle this better?
consumer_key = 'redacted'
consumer_secret = 'redacted'
access_token = 'redacted'
access_token_secret = 'redacted'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# ----- PARAMETERS -----

filename = 'outputfilename.txt'

# Add more emotion entries here
search_emojis = {
    Emotion.JOY: [':smile:', ':smiley:', ':grin:', ':grinning:'],
    # Emotion.ANGER: [':rage:', ':anger_symbol:', ':angry_face:'],
}

# Number of tweets to scrape for each emotion
num_tweets = 500

# Minimum length of tweets to consider
min_length = 50

# --- END PARAMETERS ---

search_unicodes = {k: [emoji.EMOJI_ALIAS_UNICODE[s] for s in v] for k, v in search_emojis.items()}

total_tweets = len(search_emojis) * num_tweets

# See http://stackoverflow.com/a/41422178
emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    , flags=re.UNICODE)

file = codecs.open(filename, encoding='utf-8', mode='w')

def strip_emojis(text):
    return emoji_pattern.sub(r'', text)

def count_emojis(text):
    return len(emoji_pattern.findall(text))

def tag(status):
    # Ignore retweets
    if hasattr(status, 'retweeted_status'):
        return None

    # Ignore really short tweets (probably not very useful content)
    if len(status.text) < min_length:
        return None

    # Tweets with exactly one emoji
    if count_emojis(status.text) != 1:
        return None

    # Tweets ending with exactly one emoji
    for emotion, unicodes in search_unicodes.items():
        if status.text.endswith(tuple(unicodes)):
            return emotion
    return None

def write_tweet(tag, status):
    file.write(status.id_str)
    file.write('\n')
    file.write(status.user.screen_name.encode('unicode_escape'))
    file.write('\n')
    file.write(tag.name)
    file.write('\n')
    file.write(strip_emojis(status.text).rstrip().encode('unicode_escape'))
    file.write('\n')
    file.write('\n')

class Listener(tweepy.StreamListener):
    def __init__(self):
        super(Listener, self).__init__()
        self.counts = {}
    def on_error(self, status_code):
        print "Error", status_code
        # Disconnect on error
        return False
    def on_status(self, status):
        t = tag(status)
        if t is not None:
            if not t in self.counts:
                self.counts[t] = 0
            if self.counts[t] < num_tweets:
                write_tweet(t, status)
                self.counts[t] += 1
                print t, self.counts[t]
                if sum(self.counts.values()) == total_tweets:
                    # Disconnect stream after reaching desired count
                    return False

listener = Listener()
stream = tweepy.Stream(auth=api.auth, listener=listener)

stream.filter(languages=['en'], track=list(itertools.chain(*search_unicodes.values())))

file.close()
