from flask import Flask, render_template
import os
from utility import console, Emotion

console.log()
console.h1("Initializing Server")

from classifiers import LstmClassifier
from tts import TTS

app = Flask(__name__)
lstm = LstmClassifier()
tts = TTS()

def say(text, emotion):
    output_path = "/tmp/" + tts.as_file_path(text) + ".aif"
    output_path = tts.speak(text, emotion, output_path)
    os.system("afplay '{}' || ffplay '{}' || play '{}'".format(output_path, output_path, output_path))

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/classify/<text>")
def classify(text):
    classifications = lstm.predict([text])
    emotion = Emotion(classifications[0]).name
    console.debug(text,"->",classifications[0],"->",emotion)
    say(text, Emotion(classifications[0]))
    return "{\"" + emotion + "\" : 1.0}"

console.h1("Server Ready")
app.run()
