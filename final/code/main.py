import os
import time
import numpy as np
import gzip
import pickle
from sklearn.preprocessing import LabelBinarizer

from utils import *
from knn import KNN
from linear_model import softmaxClassifier
from neural_net import NeuralNet
from svm import supportVectorMachine
import cnn


with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

X_train, y_train = train_set
X_valid, y_valid = valid_set
X_test, y_test = test_set


def run_KNN_model():
    # final model k=4
    final_model = KNN(4)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_valid)
    tr_error = np.mean(y_pred != y_valid)
    print(f"KNN k={4} Validation error: %.5f" % tr_error)
    y_pred = final_model.predict(X_test)
    test_error = np.mean(y_pred != y_test)
    print(f"KNN k={4} Test error: %.5f" % test_error)


def run_linear_model():
    # final model: softmaxClassifier with optTol=1
    model = softmaxClassifier(optTol=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    tr_error = np.mean(y_pred != y_valid)
    print(f"softmaxClassifier Validation error: %.5f" % tr_error)
    y_pred = model.predict(X_test)
    test_error = np.mean(y_pred != y_test)
    print(f"softmaxClassifier Test error: %.5f" % test_error)


def run_MLP_model():
    binarizer = LabelBinarizer()
    Y = binarizer.fit_transform(y_train)

    # final model: 2 hidden layers each 1024 hidden units
    num_layers = 2
    hidden_layer_sizes = [1024 for _ in range(num_layers)]
    model = NeuralNet(hidden_layer_sizes, learning_rate=1e-3, max_iter=1000)

    t = time.time()
    model.fit(X_train, Y)
    print("Fitting took %d seconds" % (time.time() - t))

    # save model weights
    with open('../data/2_layer_weights.pickle', 'wb') as f:
        pickle.dump(model.weights, f)

    # Compute validation and test errors
    yhat = model.predict(X_valid)
    validError = np.mean(yhat != y_valid)
    print(f"{num_layers} hidden layers Validation error = ", validError)
    yhat = model.predict(X_test)
    testError = np.mean(yhat != y_test)
    print(f"{num_layers} hidden layers Test error = ", testError)


def run_SVM_model():
    # final SVM model with hinge loss: C=400, lr=1e-4
    rate = 1e-4
    svm = supportVectorMachine(C=400, learning_rate=rate, maxEvals=1000)
    svm.fit(X_train, y_train)

    # Compute validation and test errors
    y_pred = svm.predict(X_valid)
    tr_error = np.mean(y_pred != y_valid)
    print(f"SVM lr={rate} Validation error: %.5f" % tr_error)
    y_pred = svm.predict(X_test)
    ts_error = np.mean(y_pred != y_test)
    print(f"SVM lr={rate} Test error: %.5f" % ts_error)


def run_CNN_model():
    # read trained model weights for lr=0.01
    with open(os.path.join('..', 'data', 'cnn_lr_1e-2.pkl'), 'rb') as f:
        params = pickle.load(f)

    # Compute validation and test errors
    y_pred = cnn.predict(X_valid, params)
    tr_error = np.mean(y_pred != y_valid)
    print(f"CNN lr={0.01} Validation error: %.5f" % tr_error)
    y_pred = cnn.predict(X_test, params)
    ts_error = np.mean(y_pred != y_test)
    print(f"CNN lr={0.01} Test error: %.5f" % ts_error)


def main():
    run_KNN_model()
    run_linear_model()
    run_SVM_model()
    run_MLP_model()
    run_CNN_model()


if __name__ == '__main__':
    main()



