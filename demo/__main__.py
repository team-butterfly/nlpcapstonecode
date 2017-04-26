from flask import Flask, render_template
from utility import console
from classifiers import LstmClassifier

app = Flask(__name__)
lstm = LstmClassifier()

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/classify/<text>")
def classify(text):
    console.debug("Classifying", text)
    classifications = lstm.predict([text])
    console.debug("Got classifications", classifications)
    return "{\"sadness\" : 0.4, \"joy\" : 0.6}"

if __name__ == "__main__":
    app.run()
