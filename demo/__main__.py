from flask import Flask, render_template
from utility import console, Emotion
from classifiers import LstmClassifier
import data_source

app = Flask(__name__)
lstm = LstmClassifier()

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/classify/<text>")
def classify(text):
    classifications = lstm.predict([text])
    emotion = Emotion(classifications[0]).name
    console.debug(text,"->",classifications[0],"->",emotion)

    return "{\"" + emotion + "\" : 1.0}"

if __name__ == "__main__":
    app.run()
