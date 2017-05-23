from classifiers import GloveClassifier

g = GloveClassifier("attend_hiddens2")
while True:
    sent = input("Enter sentence: ")
    y = g.predict_soft_with_attention([sent])[0]
    print(y)
