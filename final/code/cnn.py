from neural_net import NeuralNet
from sklearn.preprocessing import LabelBinarizer
import gzip
import os
import pickle

from utils import *
import numpy as np
import tqdm


class CNN(NeuralNet):
    def __init__(self, hidden_layer_sizes, learning_rate=1e-3, lammy=1, max_iter=100):
        super().__init__(hidden_layer_sizes, learning_rate, lammy, max_iter)



# adapted for single filter from https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
def convolution(image, filter, padding=0, stride=1):
    filter_x, filter_y = filter.shape  # filter dimensions
    if len(image.shape) > 2:
        _, image_x, image_y = image.shape  # image dimensions
    else:
        image_x, image_y = image.shape

    # output dimensions
    output_x = int((image_x - filter_x + padding) / stride) + 1
    output_y = int((image_y - filter_y + padding) / stride) + 1
    output = np.zeros((output_x, output_y))

    if padding != 0:
        padded = np.zeros((image_x + padding * 2, image_y + padding * 2))
        padded[padding:-padding, padding:-padding] = image      # zero padded
        image_x, image_y = padded.shape
        image = padded

    # convolve each filter over the image
    for y in range(0, image_y - filter_y, stride):
        for x in range(0, image_x - filter_x, stride):
            output[x, y] = np.sum(filter * image[x:x+filter_x, y:y+filter_y])

    return output


# adapted for single filter from https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
def backwardConvolution(dW_prev, conv, filter, stride):
    filter_x, filter_y = filter.shape  # filter dimensions
    conv_x, conv_y = conv.shape  # conv dimensions

    dW = np.zeros((conv_x, conv_y))
    dFilter = np.zeros(filter.shape)

    for y in range(0, conv_y - filter_y, stride):
        for x in range(0, conv_x - filter_x, stride):
            # loss gradient of filter (used to update the filter)
            dFilter += dW_prev[y, x] * conv[y:y + filter_y, x:x + filter_x]
            # loss gradient of the input to the convolution operation
            dW[y:y + filter_y, x:x + filter_x] += dW_prev[y, x] * filter

    return dW, dFilter

# adapted from https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
def conv(image, label, params, conv_s):
    [f1, f2, f3, w1, w2, b1, b2] = params

    ################################################
    ############## Forward Operation ###############
    ################################################
    conv1 = convolution(image, f1, conv_s)  # convolution operation
    conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

    conv2 = convolution(conv1, f2, conv_s)  # second convolution operation
    conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

    conv3 = convolution(conv2, f3, conv_s)  # second convolution operation
    conv3[conv3 <= 0] = 0  # pass through ReLU non-linearity

    dim3_x, dim3_y = conv3.shape
    fc = conv3.reshape((dim3_x * dim3_y, 1))  # flatten layer

    z = w1.dot(fc) + b1  # first dense layer
    z[z <= 0] = 0  # pass through ReLU non-linearity

    output = w2.dot(z) + b2 # second dense layer

    probs = softmax(output)  # predict class probabilities with the softmax activation function

    ################################################
    #################### Loss ######################
    ################################################

    loss = categoricalCrossEntropy(probs, label)  # categorical cross-entropy loss

    ################################################
    ############# Backward Operation ###############
    ################################################
    dout = probs - label  # derivative of loss w.r.t. final dense layer output
    dw2 = dout.dot(z.T)  # loss gradient of final dense layer weights
    db2 = np.sum(dout, axis=1).reshape(b2.shape)  # loss gradient of final dense layer biases

    dz = w2.T.dot(dout)  # loss gradient of first dense layer outputs
    dz[z <= 0] = 0  # backpropagate through ReLU
    dw1 = dz.dot(fc.T)
    db1 = np.sum(dz, axis=1).reshape(b1.shape)

    dfc = w1.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
    dconv3 = dfc.reshape(conv3.shape)  # reshape fully connected into dimensions of conv3 layer

    dconv2, df3 = backwardConvolution(dconv3, conv2, f3, conv_s) # backpropagate previous gradient through third convolutional layer.
    dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

    dconv1, df2 = backwardConvolution(dconv2, conv1, f2, conv_s)  # backpropagate previous gradient through second convolutional layer.
    dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

    dimage, df1 = backwardConvolution(dconv1, image, f1, conv_s)  # backpropagate previous gradient through first convolutional layer.

    grads = [df1, df2, df3, dw1, dw2, db1, db2]

    return grads, loss

# use CNN for predictions when given trained parameters
def predict(images, params, conv_s=1):
    [f1, f2, f3, w1, w2, b1, b2] = params

    preds = np.zeros(len(images))

    for idx, image in enumerate(images):
        image = image.reshape(28, 28)

        conv1 = convolution(image, f1, conv_s)  # convolution operation
        conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

        conv2 = convolution(conv1, f2, conv_s)  # second convolution operation
        conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

        conv3 = convolution(conv2, f3, conv_s)  # second convolution operation
        conv3[conv3 <= 0] = 0  # pass through ReLU non-linearity

        dim3_x, dim3_y = conv3.shape
        fc = conv3.reshape((dim3_x * dim3_y, 1))  # flatten layer

        z = w1.dot(fc) + b1  # first dense layer
        z[z <= 0] = 0  # pass through ReLU non-linearity

        output = w2.dot(z) + b2  # second dense layer

        probs = softmax(output)  # predict class probabilities with the softmax activation function
        preds[idx] = np.argmax(probs)

    return preds


# following functions: softmax, categoricalCrossEntropy, initializeFilter, initializeWeight, adamGD, train taken from
# https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
def softmax(raw_preds):
    '''
    pass raw predictions through softmax activation function
    '''
    output = np.exp(raw_preds) # exponentiate vector of raw predictions
    return output/np.sum(output) # divide the exponentiated vector by its sum. All values in the output sum to 1.

def categoricalCrossEntropy(probs, label):
    '''
    calculate the categorical cross-entropy loss of the predictions
    '''
    return -np.sum(label * np.log(probs)) # Multiply the desired output label by the log of the prediction, then sum all values in the vector

def initializeFilter(size, scale = 1.0):
    '''
    Initialize filter using a normal distribution with and a
    standard deviation inversely proportional the square root of the number of units
    '''
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    '''
    Initialize weights with a random normal distribution
    '''
    return np.random.standard_normal(size=size) * 0.01


def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    update the parameters through Adam gradient descnet.
    '''
    [f1, f2, f3, w1, w2, b1, b2] = params

    X = batch[:, 0:-1]  # get batch inputs
    X = X.reshape(len(batch), dim, dim)
    Y = batch[:, -1]  # get batch labels

    cost_ = 0
    batch_size = len(batch)

    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    df3 = np.zeros(f3.shape)
    dw1 = np.zeros(w1.shape)
    dw2 = np.zeros(w2.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(f3.shape)
    v4 = np.zeros(w1.shape)
    v5 = np.zeros(w2.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)

    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(f3.shape)
    s4 = np.zeros(w1.shape)
    s5 = np.zeros(w2.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)


    for i in range(batch_size):
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

        # Collect Gradients for training example
        grads, loss = conv(x, y, params, 1)
        [df1_, df2_, df3_, dw1_, dw2_, db1_, db2_] = grads

        df1 += df1_
        df2 += df2_
        df3 += df3_
        dw1 += dw1_
        dw2 += dw2_
        db1 += db1_
        db2 += db2_

        cost_ += loss

    # Parameter Update
    v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
    s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
    f1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

    v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
    f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * df3 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (df3 / batch_size) ** 2
    f3 -= lr * v3 / np.sqrt(s3 + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * dw1 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (dw1 / batch_size) ** 2
    w1 -= lr * v4 / np.sqrt(s4 + 1e-7)

    bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
    bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
    b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

    v5 = beta1 * v5 + (1 - beta1) * dw2 / batch_size
    s5 = beta2 * s5 + (1 - beta2) * (dw2 / batch_size) ** 2
    w2 -= lr * v5 / np.sqrt(s5 + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
    bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
    b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    params = [f1, f2, f3, w1, w2, b1, b2]

    return params, cost


#####################################################
##################### Training ######################
#####################################################

def train(X, y, num_classes=10, lr=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5,
          batch_size=500, num_epochs=2, save_path='cnn_lr_1e-2.pkl'):
    # Get training data
    m = 50000
    y = y.reshape(y.shape[0], 1)
    train_data = np.hstack((X, y))

    np.random.shuffle(train_data)

    ## Initializing all the parameters
    f1, f2, f3, w1, w2 = (f, f), (f, f), (f, f), (128, 361), (10, 128)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    f3 = initializeFilter(f3)
    w1 = initializeWeight(w1)
    w2 = initializeWeight(w2)

    b1 = np.zeros((w1.shape[0], 1))
    b2 = np.zeros((w2.shape[0], 1))

    params = [f1, f2, f3, w1, w2, b1, b2]

    cost = []

    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm.tqdm(batches)
        for x, batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))

    with open(save_path, 'wb') as file:
        pickle.dump(params, file)

    return cost





def main():
    with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    X_train, y_train = train_set
    X_valid, y_valid = valid_set
    X_test, y_test = test_set

    train(X_train, y_train, lr=0.01, save_path="../data/cnn_lr_1e-2.pkl")

if __name__ == '__main__':
    main()