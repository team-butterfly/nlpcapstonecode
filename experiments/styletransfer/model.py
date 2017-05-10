import argparse
import random, string
import os
import sys

import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate, Dropout
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau

import console
import conversion
from data import Data

SLICE_SIZE_TIME = 768    # size of spectrogram slices to use
SLICE_SIZE_FREQUENCY = 768

# Slice up matrices into squares so the neural net gets a consistent size for training (doesn't matter for inference)
def chop(matrix):
    slices = []

    for time in range(0, matrix.shape[1] // SLICE_SIZE_TIME - SLICE_SIZE_TIME):
        for freq in range(0, matrix.shape[0] // SLICE_SIZE_FREQUENCY):
            s = matrix[freq * SLICE_SIZE_FREQUENCY : (freq + 1) * SLICE_SIZE_FREQUENCY,
                       time * SLICE_SIZE_TIME : (time + 1) * SLICE_SIZE_TIME]
            slices.append(s)
    reference = matrix[0 : SLICE_SIZE_FREQUENCY,
                        matrix.shape[1]  - SLICE_SIZE_TIME : matrix.shape[1]]
    references = [reference] * len(slices)

    return slices, references

def loadImages(path):
    results = []
    references = []
    for dirPath, dirNames, fileNames in os.walk(path):
        for fileName in filter(lambda f : (f.endswith(".mp3") or f.endswith(".wav")) and not f.startswith("."), fileNames):
            audio, sampleRate = conversion.loadAudioFile(os.path.join(path, fileName))
            amplitude, phase = conversion.audioFileToSpectrogram(audio, 1536)
            result, reference = chop(amplitude)
            results.extend(result)
            references.extend(reference)
            console.log("loaded slices for", fileName)
    return np.array(results)[:, :, :, np.newaxis], np.array(references)[:, :, :, np.newaxis]

class TransferBot():

    def __init__(self):
        contentInput = Input(shape=(768, 768, 1), name='contentInput')
        console.debug("contentInput shape:", contentInput._keras_shape)

        content = Conv2D(64, (4, 3), strides=(2, 1), activation='relu', padding='same')(contentInput)
        content = Conv2D(64, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        content = MaxPooling2D((3,1))(content)
        content = BatchNormalization()(content)
        content = Conv2D(64, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        content = Conv2D(64, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        content = MaxPooling2D((1,2))(content)
        content = BatchNormalization()(content)
        content = Conv2D(64, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        content = Conv2D(64, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        content = MaxPooling2D((1,2))(content)
        content = BatchNormalization()(content)
        content = Conv2D(128, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        content = Conv2D(128, (4, 3), strides=(2, 1), activation='relu', padding='same')(content)
        # content is now 1 x None x 128

        styleInput = Input(shape=(768, 768, 1), name='styleInput')
        console.debug("styleInput shape:", styleInput._keras_shape)

        style = Conv2D(64, (3, 4), strides=(1, 2), activation='relu', padding='same')(styleInput)
        style = Conv2D(64, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)
        style = MaxPooling2D((1,3))(style)
        style = BatchNormalization()(style)
        style = Conv2D(64, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)
        style = Conv2D(64, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)
        style = MaxPooling2D((2,1))(style)
        style = BatchNormalization()(style)
        style = Conv2D(64, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)
        style = Conv2D(64, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)
        style = MaxPooling2D((2,1))(style)
        style = BatchNormalization()(style)
        style = Conv2D(128, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)
        style = Conv2D(128, (3, 4), strides=(1, 2), activation='relu', padding='same')(style)


        console.debug("content shape", content._keras_shape)
        console.debug("style shape", style._keras_shape)
        content = UpSampling2D((192, 1))(content)
        style = UpSampling2D((1, 192))(style)

        combined = Concatenate()([content, style])
        # combined = style
        combined = Conv2D(128, (3,3), activation='relu', padding='same')(combined)
        # reconstruction happens here
        combined = UpSampling2D((4, 4))(combined)
        combined = Conv2D(64, (3,3), activation='relu', padding='same')(combined)
        combined = Conv2D(32, (3,3), activation='relu', padding='same')(combined)
        output = Conv2D(1, (3,3), activation='relu', padding='same')(combined)
        console.debug("output shape:", output._keras_shape)

        m = Model(inputs=[styleInput, contentInput], outputs=output)
        m.compile(loss='mean_squared_error', optimizer='adam')

        self.model = m
        self.gridSize = 4

    def saveWeights(self, path):
        self.model.save_weights(path, overwrite=True)
    def loadWeights(self, path):
        self.model.load_weights(path)
    def train(self, dataPath, epochs, batch):

        console.h1("Loading images")
        vocals, references = loadImages(dataPath)
        console.log("Starting model fit")
        self.model.fit([vocals, references], vocals, batch_size=batch, epochs=epochs)
        console.log("Training completed")
    def predict(self, style, content):
        return self.model.predict([style, content])
    def stylize(self, stylePath, contentPath, outputPath):
        styleAudio, styleSampleRate = conversion.loadAudioFile(stylePath)
        contentAudio, contentSampleRate = conversion.loadAudioFile(contentPath)
        style, stylePhase = conversion.audioFileToSpectrogram(styleAudio, 1536)
        content, contentPhase = conversion.audioFileToSpectrogram(contentAudio, 1536)

        conversion.saveSpectrogram(style, stylePath + "_spectrogram.png")
        conversion.saveSpectrogram(content, contentPath + "_spectrogram.png")
        prediction = self.predict(style[np.newaxis,:768,:768,np.newaxis], content[np.newaxis,:768, :768, np.newaxis])
        result = np.zeros(content.shape)
        result[:prediction.shape[1], :prediction.shape[2]] = prediction[0,:,:,0]
        conversion.saveSpectrogram(result, contentPath + "_stylized_spectrogram.png")
        audio = conversion.spectrogramToAudioFile(result, 1536)
        conversion.saveAudioFile(audio, outputPath, 22050)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="weights.h5", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()
    console.time("creating model")
    transferbot = TransferBot()
    console.timeEnd("creating model")

    if len(args.files) == 0 and args.data:
        console.log("No files provided; attempting to train on " + args.data + "...")
        if args.load:
            console.h1("Loading Weights")
            transferbot.loadWeights(args.weights)
        console.h1("Training Model")
        transferbot.train(args.data, args.epochs, args.batch)
        transferbot.saveWeights(args.weights)
    elif len(args.files) > 0:
        console.log("Weights provided; performing inference on " + str(args.files) + "...")
        console.h1("Loading weights")
        transferbot.loadWeights(args.weights)
        for i, f in enumerate(args.files):
            if (not (i == len(args.files) - 1 and len(args.files) > 1)):
                transferbot.stylize(f, args.files[-1], f + "_stylized.wav")
    else:
        console.error("Please provide data to train on (--data) or files to infer on")
