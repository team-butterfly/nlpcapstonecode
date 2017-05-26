import numpy as np
import matplotlib
matplotlib.use("Agg") # Backend without interactive display
import matplotlib.pyplot as plt
from data_source import TweetsDataSource
from classifiers import GloveClassifier
from utility import console, Emotion

RAND = np.random.RandomState(12)
SAMPLES_PER_CLASS = 4
PLOT_COLORS = {
    Emotion.ANGER:   [1.000, 0.129, 0.345], # Red
    Emotion.SADNESS: [0.231, 0.639, 0.988], # Blue
    Emotion.JOY:     [0.671, 0.847, 0] # Light green
}

PLOT_FONT = {
    "size": 12,
    "fontname": "Monospace"
}

def plot_attention(words, attns, emotion, path):
    assert len(words) == len(attns), "Mismatched sentence and attention lengths: {}, {}".format(words, attns)

    rgb = PLOT_COLORS[emotion]

    fig, ax = plt.subplots()
    ax.axis("off")
    trans = ax.transData
    renderer = ax.figure.canvas.get_renderer()
    rgba = rgb + [0]
    width, max_w = 0, 312
    bbox = {"fc": rgba, "ec": (0, 0, 0, 0), "boxstyle": "round"} # Word bounding box
    for word, attn in zip(words, attns):
        rgba[3] = attn # Set opacity to attention weight 
        text = ax.text(0, 0.9, word, bbox=bbox, transform=trans, **PLOT_FONT)
        text.draw(renderer)
        ex = text.get_window_extent()
        if width > max_w:
            trans = matplotlib.transforms.offset_copy(text._transform, x=-width, y=-ex.height*2, units="dots")
            width = 0
        else:
            dw = ex.width + 20
            trans = matplotlib.transforms.offset_copy(text._transform, x=dw, units="dots")
            width += dw
    plt.savefig(path, transparent=True)
    plt.close(fig)


def plots():
    ds = TweetsDataSource(file_glob="data/tweets.v3.part*.txt", random_seed=5, tokenizer="word")
    glove = GloveClassifier("attention_hiddens")

    # Choose a random stratified sample of tweets
    indices = np.arange(len(ds.test_inputs))
    sample_idx = []
    for em in (Emotion.ANGER, Emotion.SADNESS, Emotion.JOY):
        pool = indices[np.equal(ds.test_labels, em.value)]
        sample_idx.extend(RAND.choice(pool, SAMPLES_PER_CLASS, replace=False))

    data = glove.predict_soft_with_attention(ds.test_inputs[sample_idx])

    i = 1
    for tokens, emos, attns in data:
        attn_clip = attns[:len(tokens)]
        em_pred = max(emos, key=emos.get) # Find Emotion with maximum probability
        fname = "plots/attention_{:02d}.png".format(i)
        plot_attention(tokens, attn_clip, em_pred, fname)
        console.info("Saved plot to", fname)
        i += 1


if __name__ == "__main__":
    plots()
