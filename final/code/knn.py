"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k, distance='Cosine'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)

        # cosine distance outperforms Euclidean
        if self.distance == 'Euclidean':
            dist2 = utils.euclidean_dist_squared(X, Xtest)
        else:
            dist2 = cosine_distance(X, Xtest)

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(dist2[:,i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat


def cosine_distance(X1, X2):
    N1, D1 = X1.shape
    N2, D2 = X2.shape

    dot = np.dot(X1, X2.T)
    norm1 = np.reshape(np.linalg.norm(X1, axis=1), (N1, 1))
    norm2 = np.reshape(np.linalg.norm(X2, axis=1), (N2, 1))
    similarity = dot / np.dot(norm1, norm2.T)
    distance = 1 - similarity
    return distance




