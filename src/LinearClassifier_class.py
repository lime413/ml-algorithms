import numpy as np


class LinearClassifier:
    def __init__(self, step=0.1, e=1e-3):
        self.weights = None
        self.step = step
        self.e = e

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, attributes, answers):
        attributes_extended = np.hstack([np.ones((attributes.shape[0], 1)), attributes])
        self.weights = np.zeros((attributes_extended.shape[1], 1))
        while np.linalg.norm(gradient := attributes_extended.T @ (self.sigmoid(attributes_extended @ self.weights) - answers) / attributes_extended.shape[0]) > self.e:
            self.weights -= self.step * gradient

    def predict(self, attributes):
        return self.sigmoid(self.weights[0] + attributes @ self.weights[1:])