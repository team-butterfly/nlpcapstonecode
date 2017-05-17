from flask import Flask, render_template
import os
import numpy as np
from utility import console, Emotion
import json

console.log()
console.h1("Initializing Server")

from classifiers import LstmClassifier
from tts import IBMTTS as TTS

app = Flask(__name__)
lstm = LstmClassifier()
tts = TTS()

def say(text, emotion):
    output_path = "/tmp/" + tts.as_file_path(text) + ".wav"
    output_path = tts.speak(text, emotion, output_path)

    os.system("ffplay '{}' || play '{}'".format(output_path, output_path, output_path))

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/classify/<text>")
def classify(text):
    classifications = lstm.predict_soft([text])[0]
    console.debug("classifications:", classifications)
    console.debug("best:", np.argmax(classifications),Emotion(np.argmax(classifications)))
    say(text, Emotion(np.argmax(classifications)))
    output = { Emotion(i).name : str(classifications[i]) for i in range(len(classifications)) }
    console.debug("output is ", output)
    return json.dumps(output)

console.h1("Server Ready")
app.run()
