
import sys
import os
import numpy as np
import h5py

import console
import conversion

SLICE_SIZE = 128    # size of spectrogram slices to use

# Slice up matrices into squares so the neural net gets a consistent size for training (doesn't matter for inference)
def chop(matrix, scale):
    slices = []
    for time in range(0, matrix.shape[1] // scale):
        for freq in range(0, matrix.shape[0] // scale):
            s = matrix[freq * scale : (freq + 1) * scale,
                       time * scale : (time + 1) * scale]
            slices.append(s)
    return slices

class Data:
    def __init__(self, inPath, fftWindowSize=1536, trainingSplit=0.9):
        self.inPath = inPath
        self.fftWindowSize = fftWindowSize
        self.trainingSplit = trainingSplit
        self.x = []
        self.y = []
        self.load()
    def train(self):
        return (self.x[:int(len(self.x) * self.trainingSplit)], self.y[:int(len(self.y) * self.trainingSplit)])
    def valid(self):
        return (self.x[int(len(self.x) * self.trainingSplit):], self.y[int(len(self.y) * self.trainingSplit):])
    def load(self, saveDataAsH5=False):
        h5Path = os.path.join(self.inPath, "data.h5")
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")
            self.x = h5f["x"][:]
            self.y = h5f["y"][:]
        else:
            count = 0
            for dirPath, dirNames, fileNames in os.walk(self.inPath):
                for fileName in filter(lambda f : (f.endswith(".mp3") or f.endswith(".wav")) and not f.startswith("."), fileNames):
                    audio, sampleRate = conversion.loadAudioFile(os.path.join(self.inPath, fileName))
                    amplitude, phase = conversion.audioFileToSpectrogram(audio, self.fftWindowSize)
                    console.bounds(amplitude, fileName + " amplitude")
                    console.bounds(phase, fileName + " phase")
                    dim = SLICE_SIZE
                    amplitudeSlices = chop(amplitude, dim)
                    phaseSlices = chop(phase, dim)
                    xSlices = []
                    ySlices = []
                    for i in range(0, len(amplitudeSlices)):
                        if (np.any(amplitudeSlices[i])):
                            xSlices.append(amplitudeSlices[i])
                            ySlices.append(phaseSlices[i])
                    count += 1
                    self.x.extend(xSlices)
                    self.y.extend(ySlices)
            console.info("Created", count, "pairs with", len(self.x), "total slices so far")
            # Add a "channels" channel to please the network
            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]
            # Save to file if asked
            if saveDataAsH5:
                h5f = h5py.File(h5Path, "w")
                h5f.create_dataset("x", data=self.x)
                h5f.create_dataset("y", data=self.y)
                h5f.close()

if __name__ == "__main__":
    # Simple testing code to use while developing
    console.h1("Loading Data")
    d = Data(sys.argv[1], 1536)
    console.h1("Writing Sample Data")
    for i in range(0,10):
        console.bounds(d.x[i][:, :, 0], "x[" + str(i) + "]")
        console.bounds(d.y[i][:, :, 0], "y[" + str(i) + "]")
        conversion.saveSpectrogram(d.x[i][:, :, 0], "x_sample_" + str(i) + ".png")
        conversion.saveSpectrogram(d.y[i][:, :, 0], "y_sample_" + str(i) + ".png")
