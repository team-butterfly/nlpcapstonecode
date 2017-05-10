import argparse
import random, string
import os

import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate, Dense, Reshape, Lambda, AveragePooling2D
from keras.models import Model
import keras.backend as K

import console
import conversion
from data import Data

class PhaseBot:
    def __init__(self):

        amplitude = Input(shape=(None, None, 1), name='input')
        noise = Input(shape=(None, None, 1), name='noise')
        genInput = Concatenate()([amplitude, noise])
        conv = Conv2D(32, 3, strides=2, activation='relu', padding='same')(genInput)
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = UpSampling2D((4,4))(conv)
        conv = Concatenate()([conv, genInput])
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(1, 3, activation='tanh', padding='same')(conv)
        phase = conv
        generator = Model(inputs=[amplitude, noise], outputs=phase)
        # generator.compile(loss='mean_absolute_error', optimizer='adam')

        self.generator = generator;

        discAmplitude = Input(shape=(None, None, 1), name='discAmplitude')
        discPhase = Input(shape=(None, None, 1), name='discPhase')
        discInput = Concatenate()([discAmplitude, discPhase])
        discConv = Conv2D(32, 4, strides=2, activation='relu', padding='same')(discInput)
        discConv = Conv2D(32, 4, strides=2, activation='relu', padding='same')(discConv)
        discConv = Conv2D(16, 4, strides=2, activation='relu', padding='same')(discConv)
        discConv = Conv2D(1, 4, strides=2, activation='relu', padding='same')(discConv)
        discOpinion = AveragePooling2D((16,16))(discConv)
        # discOpinion = Lambda(lambda xin: K.mean(xin, axis=[3,2,1]), output_shape=(1,))(discConv)

        discriminator = Model(inputs=[discAmplitude, discPhase], outputs=discOpinion)
        discriminator.compile(loss='binary_crossentropy', optimizer='adam')
        discriminator.trainable = False

        self.discriminator = discriminator

        ganInput = [amplitude, noise]
        newPhase = generator(ganInput)
        ganOutput = discriminator([amplitude, newPhase])
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer='adam')

        self.gan = gan

        self.peakDownscaleFactor = 8

    def train(self, data, epochs, batch=8):
        xTrain, yTrain = data.train()
        xValid, yValid = data.valid()
        while epochs > 0:
            console.log("Training for", epochs, "epochs on", len(xTrain), "examples")
            for i in range(0, epochs):
                batchIndices = np.random.randint(0, xTrain.shape[0], size=batch)
                amps = xTrain[batchIndices]
                y = yTrain[batchIndices]

                refPhase = np.concatenate([y,y])

                noise = np.random.normal(0, 1, size=[batch, amps[0].shape[0], amps[0].shape[1], 1])
                yFakes = self.generator.predict([amps, noise])

                x = np.concatenate([amps, amps])
                y = np.concatenate([y, yFakes])
                z = np.zeros((len(x)))
                z[:batch] = 0.9

                z = z[:, np.newaxis, np.newaxis, np.newaxis]

                self.discriminator.trainable = True
                dLoss = self.discriminator.train_on_batch([x, y], z)
                noise = np.random.normal(0, 1, size=[batch, amps[0].shape[0], amps[0].shape[1], 1])
                zGoal = np.ones(batch)
                zGoal = zGoal[:, np.newaxis, np.newaxis, np.newaxis]
                self.discriminator.trainable = False
                gLoss = self.gan.train_on_batch([amps, noise], zGoal)

                console.bounds(y[0], "real sample")
                console.bounds(yFakes[0], "fake sample")
            for j in range(0, batch):
                conversion.saveSpectrogram(y[j][:,:,0], str(j) + "_real.png")
                conversion.saveSpectrogram(yFakes[j][:,:,0], str(j) + "_fake.png")
            console.info(i, "dLoss:", dLoss,"gLoss",gLoss)

            # self.generator.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))


            console.notify(str(epochs) + " Epochs Complete!", "Training on", data.inPath, "with size", batch)
            while True:
                try:
                    epochs = int(input("How many more epochs should we train for? "))
                    break
                except ValueError:
                    console.warn("Oops, number parse failed. Try again, I guess?")
            if epochs > 0:
                save = input("Should we save intermediate weights [y/n]? ")
                if not save.lower().startswith("n"):
                    weightPath = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
                    console.log("Saving intermediate weights to", weightPath)
                    self.saveWeights(weightPath)


    def saveWeights(self, path):
        self.generator.save_weights(path, overwrite=True)
    def loadWeights(self, path):
        self.generator.load_weights(path)
    def reconstructVocals(self, path, fftWindowSize):
        console.log("Attempting to reconstruct vocals from", path)
        spectrogram, sampleRate = conversion.loadSpectrogram(path)

        expandedSpectrogram = conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)
        expandedSpectrogramWithBatchAndChannels = expandedSpectrogram[np.newaxis, :, :, np.newaxis]
        noise = np.random.normal(0, 1, size=expandedSpectrogramWithBatchAndChannels.shape)
        predictedPhaseWithBatchAndChannels = self.generator.predict([expandedSpectrogramWithBatchAndChannels, noise])
        predictedPhase = predictedPhaseWithBatchAndChannels[0, :, :, 0] # o /// o
        newPhase = np.clip(predictedPhase[:spectrogram.shape[0], :spectrogram.shape[1]] * 2, -.99, .99)
        console.bounds(newPhase, "newPhase")
        console.log("Processed spectrogram; reconverting to audio")

        newAudio = conversion.spectrogramToAudioFile(spectrogram, fftWindowSize=fftWindowSize, phase=newPhase)
        console.bounds(newAudio, "newAudio")
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])
        outputFileNameBase = os.path.join(pathParts[0], fileNameParts[0] + "_reconstructed")
        console.log("Converted to audio; writing to", outputFileNameBase)

        conversion.saveSpectrogram(spectrogram, os.path.join(pathParts[0], fileNameParts[0]) + ".png")
        conversion.saveSpectrogram(newPhase, outputFileNameBase + ".png")
        conversion.saveAudioFile(newAudio, outputFileNameBase + ".wav", sampleRate)
        console.log("Phase reconstruction complete 👌")

# train on amplitude, phase pairs

if __name__ == "__main__":
    # if data folder is specified, create a new data object and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="weights.h5", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()
    phaseBot = PhaseBot()

    if len(args.files) == 0 and args.data:
        console.log("No files provided; attempting to train on " + args.data + "...")
        if args.load:
            console.h1("Loading Weights")
            phaseBot.loadWeights(args.weights)
        console.h1("Loading Data")
        data = Data(args.data, args.fft, args.split)
        console.h1("Training Model")
        phaseBot.train(data, args.epochs, args.batch)
        phaseBot.saveWeights(args.weights)
    elif len(args.files) > 0:
        console.log("Weights provided; performing inference on " + str(args.files) + "...")
        console.h1("Loading weights")
        phaseBot.loadWeights(args.weights)
        for f in args.files:
            phaseBot.reconstructVocals(f, args.fft)
    else:
        console.error("Please provide data to train on (--data) or files to infer on")
