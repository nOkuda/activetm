import numpy as np


def pR2(model, words, labels, labelmean):
    totalss = np.power(np.subtract(labels, labelmean), 2).sum()
    predictions = []
    for word in words:
        predictions.append(model.predict(word))
    residual = np.power(np.subtract(labels, predictions), 2).sum()
    return 1 - (residual / totalss)

