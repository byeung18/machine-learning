import numpy as np
from numpy.linalg import solve, norm
from scipy.optimize import approx_fprime
import utils

class softmaxClassifier:
    def __init__(self, maxEvals=500, optTol=1e-2, gamma=1e-4, verbose=0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.optTol = optTol
        self.gamma = gamma

    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes
        W = np.reshape(w, (k, d))

        XW = np.dot(X, W.T)
        K = np.max(XW, axis=1).reshape(n, 1)
        XW -= K
        M = np.sum(np.exp(XW), axis=1)

        y_bin = np.zeros((n, k)).astype(bool)
        y_bin[np.arange(n), y] = 1

        f = -np.sum(XW[y_bin] - np.log(M))
        grad = ((np.exp(XW)) / M[:, None] - y_bin).T@X
        # print(f, grad.flatten().shape)
        return f, grad.flatten()

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        k = self.n_classes
        # Initial guess
        self.w = np.zeros(d*k)
        # self.w = self.W
        (self.w, f) = findMin(self.funObj, self.w,
                                      self.maxEvals, self.optTol, self.gamma, X, y, verbose=self.verbose)
        # utils.check_gradient(self, X, y)

        self.w = np.reshape(self.w, (k, d))

    def predict(self, X):
        return np.argmax(X@self.w.T, axis=1)


def findMin(funObj, w, maxEvals, optTol, gamma, *args, verbose=0):
    """
    Uses gradient descent to optimize the objective function

    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    # optTol = 1e-2
    # gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1

    alpha = 1.
    while True:
        # Line-search using quadratic interpolation to find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1

            if f_new <= f - gamma * alpha*gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Print progress
        if verbose > 0:
            print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        if verbose > 1:
            print("alpha: %.3f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, f