#!/usr/bin/env python3

import codecs
import emoji
import itertools
import json
import re
import sys
import tweepy

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

filename = sys.argv[1]

# Add more emotion entries here
search_emojis = {
    Emotion.JOY: [':grinning:', ':grin:', ':joy:', ':rolling_on_the_floor_laughing:', ':smiley:', ':smile:', ':party_popper:'],
    Emotion.SADNESS: [':frowning_face:', ':slightly_frowning_face:', ':disappointed:', ':crying_face:', ':loudly_crying_face:', ':neutral_face:'],
    Emotion.ANGER: [':rage:', ':anger_symbol:', ':angry_face:'],
}

# Maximum number of tweets to scrape for each emotion
num_tweets = 1000

# Minimum length of tweets to consider
min_length = 50

# --- END PARAMETERS ---

search_unicodes = {k: [emoji.EMOJI_ALIAS_UNICODE[s] for s in v] for k, v in search_emojis.items()}

total_tweets = len(search_emojis) * num_tweets

# See http://stackoverflow.com/a/41422178
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]", flags=re.UNICODE)

file = codecs.open(filename, mode='w')#, encoding='utf-8')
file.write("v.4/21\n")

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
    d = {
        'id': status.id_str,
        'time': str(status.created_at),
        'username': status.user.screen_name,
        'tag': tag.name,
        'origtext': status.text,
        'text': strip_emojis(status.text).rstrip(),
    }
    file.write(json.dumps(d))
    file.write('\n')
    file.flush()

class Listener(tweepy.StreamListener):
    def __init__(self):
        super(Listener, self).__init__()
        self.counts = {}
    def on_error(self, status_code):
        print("Error", status_code)
        # Disconnect on error
        return False
    def on_status(self, status):
        t = tag(status)
        if t is not None:
            if not t.name in self.counts:
                self.counts[t.name] = 0
            if self.counts[t.name] < num_tweets:
                write_tweet(t, status)
                self.counts[t.name] += 1
                print(self.counts)
                if sum(self.counts.values()) == total_tweets:
                    # Disconnect stream after reaching desired count
                    return False

listener = Listener()
stream = tweepy.Stream(auth=api.auth, listener=listener)

stream.filter(languages=['en'], track=list(itertools.chain(*search_unicodes.values())))

file.close()
