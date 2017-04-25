class _Emotion():
    NEUTRAL = 0
    JOY = 1
    SADNESS = 2
    ANGER = 3
    DISGUST = 4
    SURPRISE = 5

    def __getitem__(self, item):
        return getattr(self, item)

Emotion = _Emotion()
