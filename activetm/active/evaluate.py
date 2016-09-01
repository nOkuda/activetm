import numpy as np


def get_predictions(model, words):
    """Get predictions on words from model

    Predictions are ordered as words is
    """
    predictions = []
    for word in words:
        predictions.append(model.predict(word))
    return predictions


def pR2(predictions, labels, labelmean):
    """Calculate predictive R^2"""
    totalss = np.power(np.subtract(labels, labelmean), 2).sum()
    residual = np.power(np.subtract(labels, predictions), 2).sum()
    return 1 - (residual / totalss)

def mean_absolute_errors(predictions, labels):
    """Calculate mean absolute errors"""
    return np.fabs(np.asarray(predictions) - np.asarray(labels))
