from tts import IBMTTS
ibm = IBMTTS()
from utility import Emotion
ibm.speak(text="kittens are pretty cute", emotion=Emotion.JOY, output_path="/Users/ollin/desktop/test_joy.wav")
ibm.speak(text="kittens are pretty cute", emotion=Emotion.SADNESS, output_path="/Users/ollin/desktop/test_sadness.wav")
ibm.speak(text="kittens are pretty cute", emotion=Emotion.ANGER, output_path="/Users/ollin/desktop/test_anger.wav")
