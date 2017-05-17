from flask import Flask, render_template
import os
import numpy as np
from utility import console, Emotion

console.log()
console.h1("Initializing Server")

from classifiers import LstmClassifier
from tts import IBMTTS as TTS

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
    classifications = lstm.predict_soft([text])[0]
    console.debug("classifications:", classifications)
    console.debug("best:", np.argmax(classifications),Emotion(np.argmax(classifications)))
    say(text, Emotion(np.argmax(classifications)))
    output = "{"
    for i in range(0, len(classifications)):
        emotion = Emotion(i).name
        weight = classifications[i]
        if i > 0:
            output += (",\n")
        output += "\"" + str(emotion) + "\" : " + str(weight)
    return output + "}"

console.h1("Server Ready")
app.run()
