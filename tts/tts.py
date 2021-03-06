import os
import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
import re
import requests
import json
from utility import console
import threading
import subprocess
from aeneas.tools.execute_task import ExecuteTaskCLI


import tts.credentials as credentials
from utility import console, Emotion

class TTS():
    envelopes = {
        Emotion.NEUTRAL : [
            (0.0,  0.0),
            (1.0,  0.0)
        ],
        Emotion.JOY : [
            (0.0,  0.0),
            (0.4,  0.05),
            (0.6,  0.1),
            (0.7,  0.1),
            (0.9,  0.15),
            (1.0,  0.2)
        ],
        Emotion.ANGER : [
            (0.0,  0.0),
            (0.4, -0.1),
            (0.6, -0.1),
            (0.7,  0.0),
            (0.8, -0.1),
            (1.0, -0.2)
        ],
        Emotion.SURPRISE : [
            (0.0,  0.0),
            (0.7, 0.0),
            (0.8, 0.1),
            (1.0, 0.2)
        ],
        Emotion.SADNESS : [
            (0.0,  0.0),
            (0.5, -0.1),
            (1.0, -0.2)
        ]
    }
    def __init__(self):
        pass

    def get_envelope_value(self, envelope, x):
        assert 0 <= x <= 1
        prev_pt = [pair for pair in envelope if pair[0] <= x][-1]
        next_pt = [pair for pair in envelope if pair[0] > x][0]
        assert prev_pt[0] <= x <= next_pt[0]
        progress = (x - prev_pt[0])/(next_pt[0] - prev_pt[0])
        val = progress * next_pt[1] + (1 - progress) * prev_pt[1]
        return val

    def file_to_spectrogram(self, file_path):
        audio, sample_rate = librosa.load(file_path)
        spectrogram = librosa.stft(audio, 1536)
        phase = np.imag(spectrogram)
        amplitude = np.log1p(np.abs(spectrogram))
        return amplitude, phase

    def spectrogram_to_file(self, spectrogram, phase, file_path):
        amplitude = np.exp(spectrogram) - 1
        for i in range(10):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * (2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, 1536)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)

        librosa.output.write_wav(file_path, audio, 22050, norm=True)

    def save_spectrogram(self, spectrogram, filePath):
        spectrum = spectrogram
        image = np.clip((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)), 0, 1)
        # Low-contrast image warnings are not helpful, tyvm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(filePath, image)

    def apply_pitch_envelope_to_matrix(self, matrix, envelope):
        original = matrix
        new = np.zeros(matrix.shape)
        for t in range(0, matrix.shape[1]):
            oldcol = original[:, t:t+1]
            stretch_factor = (1 + self.get_envelope_value(envelope, t / matrix.shape[1]))
            # console.info(t, ":", stretch_factor)
            newheight = round(oldcol.shape[0] * stretch_factor)
            newcol = scipy.misc.imresize(oldcol, (newheight, 1)).astype(float)
            newcol *= oldcol.max() / (newcol.max() + 0.1)
            clipped = newcol[:new.shape[0], 0]
            new[:clipped.shape[0], t] = clipped

        return new

    def apply_pitch_envelope_to_file(self, file_path, envelope):
        # get spectrogram
        # console.debug("file_to_spectrogram", file_path)
        spectrogram, phase = self.file_to_spectrogram(file_path)
        # self.save_spectrogram(spectrogram, "/Users/ollin/Desktop/original.png")
        # apply envelope
        spectrogram2 = self.apply_pitch_envelope_to_matrix(spectrogram, envelope)
        # self.save_spectrogram(spectrogram2, "/Users/ollin/Desktop/distorted.png")
        # save output
        new_file_path = file_path.replace(".aif", ".wav")
        # console.debug("spectrogram_to_file new_file_path", new_file_path)
        self.spectrogram_to_file(spectrogram2, np.zeros(phase.shape), new_file_path)
        os.remove(file_path)
        return new_file_path

    def speak_raw(self, text, output_path):
        # console.debug("speak_raw output_path", output_path)
        assert output_path.endswith(".aif") or output_path.endswith(".aiff")
        os.system("say '" + text.replace("'", "") + "' -o '" + output_path + "'")

    def speak(self, text, emotion, output_path):
        # console.debug("speak output_path", output_path)
        self.speak_raw(text, output_path)
        new_file_path = self.apply_pitch_envelope_to_file(output_path, self.envelopes[emotion])
        # console.debug("speak new_file_path", new_file_path)
        console.log("Spoke output to", new_file_path)
        return new_file_path
    def as_file_path(self, text):
        return re.compile('[^a-z_]+').sub('', text.lower().replace(" ", "_"))

class IBMTTS(TTS):
    def __init__(self):
        # load IBM web interface
        self.emotionsToIBMTypes = {
            Emotion.JOY : "GoodNews",
            Emotion.ANGER : "Uncertainty",
            Emotion.SADNESS : "Apology"
        }
    def speak(self, text, emotion, output_path):
        if emotion in self.emotionsToIBMTypes:
            annotatedText = '<express-as type="' + self.emotionsToIBMTypes[emotion] + '">' + text + '</express-as>'
        else:
            annotatedText = text
        params = {
            'text' : annotatedText,
            'voice' : 'en-US_AllisonVoice',
            'accept' : 'audio/ogg',
        }
        console.debug("sending params", params)
        baseURL = "https://stream.watsonplatform.net/text-to-speech/api/v1/synthesize"

        response = requests.get(baseURL, params=params, auth=(credentials.USERNAME, credentials.PASSWORD), verify=False, stream=True)
        # query IBM web interface

        with open(output_path, "wb") as output:
            for data in response.iter_content(2048):
                output.write(data)
        console.debug("Wrote to", output_path)
        return output_path
    def modulate(self, audio_path, text_path, attention):
        json_path = text_path + ".json"
        def runCli():
            cli = ExecuteTaskCLI()
            cli.run(arguments=["", audio_path, text_path, "task_language=eng|os_task_file_format=json|is_text_type=plain", json_path])
        task = threading.Thread(target=runCli)
        task.start()
        task.join()

        audio, sample_rate = librosa.load(audio_path, mono=True)
        output_audio_path = audio_path + ".wav"

        librosa.output.write_wav(audio_path + "_unmodulated.wav", audio, sample_rate, norm=True)
        with open(json_path) as mapping_file:
            mapping = json.load(mapping_file)
        word_positions = [(float(x["begin"]), float(x["end"])) for x in mapping["fragments"]]
        total_time = audio.shape[0] / sample_rate
        for i in range(0, len(word_positions)):
            word_position = word_positions[i]
            scale = 2 * attention[i] / sum(attention) + 1.0
            start = int(word_position[0] / total_time * audio.shape[0])
            end = int(word_position[1] / total_time * audio.shape[0])
            audio[start:end] *= scale
        librosa.output.write_wav(output_audio_path, audio, sample_rate, norm=True)
        console.info("Transcoding to ogg")
        subprocess.call(["ffmpeg","-y", "-i", output_audio_path, audio_path])
        console.info("Wrote modulated audio to", audio_path)
        return audio_path
    def speak_with_modulation(self, text, emotion, output_path, attention):
        output_path = self.speak(text, emotion, output_path)
        text_file_path = "/tmp/" + self.as_file_path(text) + ".txt"
        with open(text_file_path, "w") as text_file:
            text_file.write("\n".join(text.split()))
        output_path = self.modulate(output_path, text_file_path, attention)
        return output_path
