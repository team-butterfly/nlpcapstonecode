import numpy as np
import argparse
import skimage.io as io
from os.path import basename
import random
from model import TransferBot

import console

def loadImage(path):
    console.info("Loading from", path)
    return io.imread(path, as_grey=True)

def saveImage(image, path):
    console.info("Saving to", path)
    io.imsave(path, image)

# OSE 0.314 OSL 52 OW 0.369 OS 0.91
# OSE 0.023 OSL 78 OW 0.539 OS 0.195
OUTPUT_SMOOTHING_ERROR, OUTPUT_SMOOTHING_LENGTH, OVERLAP_WEIGHT, OUTPUT_SMOOTHING = (0.1, 20, 0.5, 0.2)

# i1 is the reference, i2 is the candidate
def error(i1, i2, errorWeights):
    if (i1.shape != i2.shape):
        # lol code
        return 100
    if (len(i1.shape) == 1):
        err = OVERLAP_WEIGHT * np.dot(np.clip(i1 - i2 ,0, 1), errorWeights) + (1-OVERLAP_WEIGHT) * np.dot(np.abs(i1 - i2), errorWeights)
    else:
        err = OVERLAP_WEIGHT * np.dot(np.average(np.clip(i1 - i2, 0, 1), -1), errorWeights) + (1-OVERLAP_WEIGHT) * np.dot(np.average(np.abs(i1 - i2), -1), errorWeights)
    return 100 * err

# i1 is the reference, i2 is the candidate, i3 is the old candidate
def weightedError(i1, i2, i3, errorWeights):
    return (1-OUTPUT_SMOOTHING_ERROR) * error(i1, i2, errorWeights) + OUTPUT_SMOOTHING_ERROR * error(i3, i2, np.zeros(i2.shape) + .1)
def audioEncoder(style, content):
    model = TransferBot()
    return model.predict(content, style)
def audioPatches(style, content, maxFreqShift=128, numPatches=100):
    # setup
    errorWeights = np.asarray([1 / (i + 1)**3 for i in range(0, style.shape[0])])
    # normalize
    style = style.astype(float) / style.max()
    content = content.astype(float) / content.max()
    console.info("running audio patches algorithm")
    transferred = np.zeros(content.shape)

    # hacked together algorithm
    console.debug("error of zeros is", error(content, transferred, errorWeights))
    for t in range(0, content.shape[1]):
        original = content[:, t]
        best = transferred[:, t]
        bestScore = weightedError(original, best, best, errorWeights)
        # try some columns to try and find a better one
        for i in range(0, numPatches):
            i = random.randint(0, style.shape[1] - 1)
            for j in range(0, 64):
                j = random.randint(0, maxFreqShift) - maxFreqShift // 2
                l = style.shape[0]
                a = max(0, j)
                b = min(l + j, l)
                c = max(-j, 0)
                d = min(l - j, l)
                assert d - c == b - a
                candidate = np.zeros((l))
                candidate[a : b] = style[c : d,i]
                candidateScore = weightedError(original, candidate, transferred[:, t], errorWeights)
                if candidateScore < bestScore:
                    bestScore = candidateScore
                    best = candidate
                    offset = 1 if ((t + OUTPUT_SMOOTHING_LENGTH + 1) >= content.shape[1] or (i + OUTPUT_SMOOTHING_LENGTH + 1) >= style.shape[1]) else OUTPUT_SMOOTHING_LENGTH
                    # console.debug(offset)
                    transferred[a : b, t:t+offset] = transferred[a : b, t:t+offset] * OUTPUT_SMOOTHING + (1 - OUTPUT_SMOOTHING) * style[c : d,i:i+offset]
            # (a,b,c,d,j) = best
            # e = 0
            # while (i+e) < content.shape[0] and (j+e) < style.shape[1] and error(content[a:b,i:i+e], style[c:d, j:j+e], errorWeights[a:b]) < error(content[a:b,i+e], transferred[a:b,i+e], errorWeights[a:b]):
            #     e += 1

    console.debug("error of reconstruction is", error(content, transferred, errorWeights))
    return np.clip(transferred, 0, 1)

if __name__ == "__main__":
    # parse args
    # load style image
    # load content image
    # both should have same height
    # run audiopatches to magically create a fused image
    # save output image

    # later, we can rewrite this so that it operates on mp3 files instead using the conversion thing
    parser = argparse.ArgumentParser(description="style transfer for human voices")
    parser.add_argument("--trials", default=0, type=int, help="number of random hyperparameters to try")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()
    files = args.files

    if (len(files ) < 2):
        console.error("need to specify at least a style and content file")
    else:
        style = loadImage(files[0])
        console.log("Loaded style", files[0])
        for i in range(1, len(files)):
            content = loadImage(files[i])
            console.log("Loading content", files[i])
            transferred = audioPatches(style, content)
            transferredFileName = basename(files[i]) + "(In style of" + basename(files[0]) + ").png"
            saveImage(transferred, transferredFileName)
            console.log("Saved", transferredFileName)
            for j in range(0, args.trials):
                OUTPUT_SMOOTHING_ERROR, OUTPUT_SMOOTHING_LENGTH, OVERLAP_WEIGHT, OUTPUT_SMOOTHING = (random.random(), random.randint(4,80), random.random(), random.random())
                transferred = audioPatches(style, content)
                transferredFileName = basename(files[i]) + "(In style of " + basename(files[0]) + ")" + " OSE " + str(round(OUTPUT_SMOOTHING_ERROR, 3)) + " OSL " + str(OUTPUT_SMOOTHING_LENGTH) + " OW " + str(round(OVERLAP_WEIGHT, 3)) + " OS " + str(round(OUTPUT_SMOOTHING, 3)) + ".png"
                saveImage(transferred, transferredFileName)
                console.log("Saved", transferredFileName)
