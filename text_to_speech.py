# main file that runs all the things
import argparse
import os
from utility import console
from classifiers import BestClassifier
from tts import TTS

def text_to_speech(text, output_path):
    model = BestClassifier()
    tts = TTS()
    emotion = model.predict(text)
    tts.speak(text, emotion, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion-aware text-to-speech")
    parser.add_argument("text", nargs="*", default=[])

    args = parser.parse_args()


    if len(args.text) == 0:
        text = "Emotion aware text to speech is really great! Thank you for trying it out."
    else:
        text = " ".join(args.text)

    output_path = os.path.join("tmp", " ".join(text.split()))
    text_to_speech(text, output_path)
    console.log("Writing to", output_path)
    os.system("afplay '{}' || ffplay '{}' || play '{}'".format(output_path, output_path, output_path))
