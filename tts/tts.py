import os
import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
import re

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
