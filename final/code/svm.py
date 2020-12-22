import numpy as np
from sklearn.utils import shuffle

class supportVectorMachine:
    def __init__(self, maxEvals=500, C=10, learning_rate=1e-3, verbose=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.learning_rate = learning_rate
        self.lammy = 1
        self.C = C

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        self.W = np.zeros((self.n_classes, d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y != i] = -1
            ytmp[y == i] = 1
            (self.W[i], f) = self.findMin(self.W[i], X, ytmp, rate=self.learning_rate, maxEvals=self.maxEvals, verbose=self.verbose)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

    def hinge_subgradient(self, w, X, y, rate):
        # my implementation of stochastic sub-gradient descent for hinge loss
        # https://svivek.com/teaching/lectures/slides/svm/svm-sgd.pdf slide 61
        n = X.shape[0]
        for idx in range(n):
            x_i = X[idx]
            y_i = y[idx]
            if (y_i * w.T.dot(x_i) <= 1):
                w = (1 - rate) * w + rate * self.C * y_i * x_i
            else:
                w = (1 - rate) * w
        return w

    def findMin(self, w, *args, rate, maxEvals, verbose=1):
        # Evaluate the initial function value and gradient
        X, y = args
        count = 0
        while True:
            count += 1
            alpha = rate / count        # decaying learning rate
            X, y = shuffle(X, y)        # shuffle at start of every epoch
            w = self.hinge_subgradient(w, X, y, alpha)

            if count > maxEvals:
                if verbose:
                    print("Reached maximum number of function evaluations %d" % maxEvals)
                break

        return w, None
