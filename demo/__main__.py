from flask import Flask, render_template, url_for, send_file
import os
import numpy as np
import json
import datetime

from utility import console, Emotion

console.log()
console.h1("Initializing Server")

from classifiers import CustomVocabClassifier as Classifier
from tts import IBMTTS as TTS

app = Flask(__name__)
lstm = Classifier("try1")
tts = TTS()

def say(text, emotion, attention=None):
    output_path = "demo/static/audio/" + tts.as_file_path(text) + datetime.datetime.now().strftime("%s") + ".ogg"
    if attention is None:
        output_path = tts.speak(text, emotion, output_path)
    else:
        output_path = tts.speak_with_modulation(text, emotion, output_path, attention)
    return output_path
    # os.system("ffplay '{}' || play '{}'".format(output_path, output_path, output_path))

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/audio/<f>")
def audio(f):
    path = "demo/static/audio/" + f
    console.debug("path is", path)
    audioFile = open(path, "rb")
    response = send_file(audioFile, mimetype="audio/ogg", conditional=True)
    console.debug("response",dir(response))
    return response

@app.route("/classify/<text>")
def classify(text):
    console.log("text is ", text)
    tokens, classifications, attention = lstm.predict_soft_with_attention([text])[0]
    console.log("tokens are", tokens, "attention is", attention)
    console.debug("classifications:", classifications)
    maxEmotion = max(classifications, key=classifications.get)
    console.debug("best:", maxEmotion)
    output_path = say(text, maxEmotion, attention)
    output_classifications = { i.name : str(classifications[i]) for i in classifications }
    output = {
        "tokens" : tokens,
        "classifications" : output_classifications,
        "attention" : attention,
        "audio_path" : url_for('static', filename=os.path.relpath(output_path, "demo/static/"))
    }
    console.debug("output is ", output)
    return json.dumps(output)

console.h1("Server Ready")
app.run(port=8000, host="0.0.0.0")
