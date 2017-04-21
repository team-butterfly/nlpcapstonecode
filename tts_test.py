from tts import TTS
from utility import console, Emotion

if __name__ == "__main__":
    console.h1("testing tts")
    tts = TTS()
    tts.speak("Well done.", Emotion.JOY, "/Users/ollin/Desktop/1.aif")
    tts.speak("Here come the test results.", Emotion.JOY, "/Users/ollin/Desktop/2.aif")
    tts.speak("You are a horrible person.", Emotion.ANGER, "/Users/ollin/Desktop/3.aif")
    tts.speak("I'm serious, that's what it says.", Emotion.ANGER, "/Users/ollin/Desktop/4.aif")
    tts.speak("We weren't even testing for that...'", Emotion.JOY, "/Users/ollin/Desktop/5.aif")
