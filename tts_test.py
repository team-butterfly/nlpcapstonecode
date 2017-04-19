from tts import TTS
from utility import console, Emotion

if __name__ == "__main__":
    console.h1("testing tts")
    tts = TTS()
    tts.speak("I love programming so much", Emotion.JOY, "/Users/ollin/Desktop/happy.aif")
    tts.speak("But I hate debugging...", Emotion.ANGER, "/Users/ollin/Desktop/anger.aif")
