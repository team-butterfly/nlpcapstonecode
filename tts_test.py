from tts import TTS
from utility import console, Emotion

if __name__ == "__main__":
    console.h1("testing tts")
    tts = TTS()
    tts.speak("What a great day, sun is shining and I am smiling", Emotion.JOY, "/Users/ollin/Desktop/happy.aif")
    tts.speak("Weekend just got even worse. I hate all of you.", Emotion.ANGER, "/Users/ollin/Desktop/anger.aif")
