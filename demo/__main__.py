
from utility import console
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route("/classify/<text>")
def classify(text):
    return "{\"sadness\" : 0.4, \"joy\" : 0.6}"

if __name__ == "__main__":
    app.run()
