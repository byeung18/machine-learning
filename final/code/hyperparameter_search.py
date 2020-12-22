import os
import numpy as np
from utils import *
from knn import KNN
from linear_model import softmaxClassifier
from svm import supportVectorMachine
import gzip
import pickle
from sklearn.preprocessing import LabelBinarizer
from neural_net import NeuralNet
import time
import cnn

with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

X_train, y_train = train_set
X_valid, y_valid = valid_set
X_test, y_test = test_set


def run_KNN_hypersearch():
    # hyperparameter search for best k and best distance metric
    for k in range(1, 20):
        for distance in ['Euclidean', 'Cosine']:
            model = KNN(k, distance=distance)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            tr_error = np.mean(y_pred != y_valid)
            print(f"KNN k={k} {distance} Validation error: %.5f" % tr_error)


def run_linear_hypersearch():
    # optimization parameter search: optTol=1 works best
    best_error = 9999
    best_params = None
    for gamma in [1e-5, 1e-4, 1e-3, 1e-2]:
        for optTol in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
            model = softmaxClassifier(optTol=optTol, gamma=gamma)
            model.fit(X_train[:1000], y_train[:1000])
            y_pred = model.predict(X_valid)
            tr_error = np.mean(y_pred != y_valid)
            print(f"gamma={gamma}, optTol={optTol}, Validation error: %.5f" % tr_error)
            if tr_error < best_error:
                best_error = tr_error
                best_params = [gamma, optTol]
    best_gamma, best_optTol = best_params
    print(f"best hyperparameters: gamma={best_gamma}, optTol={best_optTol}")



def run_MLP_hypersearch():
    binarizer = LabelBinarizer()
    Y = binarizer.fit_transform(y_train)

    # find optimal number of hidden units
    for size in [32, 64, 128, 256, 512, 1024, 2048]:
        hidden_layer_sizes = [size]
        model = NeuralNet(hidden_layer_sizes, learning_rate=5e-4, max_iter=200)

        t = time.time()
        model.fit(X_train, Y)
        print("Fitting took %d seconds" % (time.time() - t))

        # Compute training error
        yhat = model.predict(X_valid)
        validError = np.mean(yhat != y_valid)
        print(f"{size} hidden units, Validation error = ", validError)

    # testing multiple hidden layers
    for num_layers in range(1, 4):
        hidden_layer_sizes = [64 for _ in range(num_layers)]
        model = NeuralNet(hidden_layer_sizes, learning_rate=3e-3, max_iter=500)

        t = time.time()
        model.fit(X_train, Y)
        print("Fitting took %d seconds" % (time.time() - t))

        # Compute validation error
        yhat = model.predict(X_valid)
        validError = np.mean(yhat != y_valid)
        print(f"{num_layers} hidden layers Validation error = ", validError)


def run_SVM_hypersearch():
    # finding best C: found C=400 works best
    for C in [200, 400, 600, 800, 1000]:
        svm = supportVectorMachine(C=C, learning_rate=1e-3, maxEvals=100)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_valid)
        print(y_pred)
        print(y_valid)
        tr_error = np.mean(y_pred != y_valid)
        print(f"SVM C={C} Validation error: %.5f" % tr_error)

    # look for best learning rate
    for rate in [1e-2, 1e-3, 1e-4]:
        svm = supportVectorMachine(C=400, learning_rate=rate, maxEvals=1000)
        svm.fit(X_train, y_train)

        # Compute validation error
        y_pred = svm.predict(X_valid)
        print(y_pred)
        print(y_valid)
        tr_error = np.mean(y_pred != y_valid)
        print(f"SVM lr={rate} Validation error: %.5f" % tr_error)


def run_CNN_hypersearch():
    # find optimal learning rate for CNN
    for rate in [1e-1, 1e-2, 1e-3, 1e-4]:
        cnn.train(X_train, y_train, lr=rate, save_path=f"../data/cnn_lr_{rate}.pkl")


def main():
    run_KNN_hypersearch()
    run_linear_hypersearch()
    run_SVM_hypersearch()
    run_MLP_hypersearch()
    run_CNN_hypersearch()


if __name__ == '__main__':
    main()



