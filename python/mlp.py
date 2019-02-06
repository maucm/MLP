""" building a Multi-layer perceptron with feed-fowarding and backpropagation """

import numpy as np
import pandas as pd
import sys

# MLP - Multilayer Perceptron


class MLP():
    """ A multilayer perceptron
            Args:
                    n_output: `int` number of output layers.
                    n_features: `int` number of input units.
                    hidden: `int` number of hidden layers.
                    l2: `float` regularization strength for l2 regularization.
                    l1: `float` regularization strength for l1 regularization.
                    epochs: `int` number of passes, steps.
                    eta: `eta` learning rate
                    alpha: `int` for momentum learning to add a factor of the previous gradient
                    to the weight update for faster learning.
                    decrease_constant: `float` the decrease constant for an adaptive learning rate
                    that decreases over time for better convergence.better
                    shuffle: `bool` randomize data to avoid overfitting and help with generalization.
                    minibatches: `int` splitting the data into different mini-batches for each step.
                    random_state: `None` the seed by how much we randomize data. Makes the random numbers predictable.
    """

    def __init__(self, n_output, n_features, hidden, l1, l2, epochs, eta, alpha, decrease_const, minibatches, shuffle=True, random_state=None):
        np.random.seed(random_state)
        self.n_output = n_output
        self.n_features = n_features
        self.hidden = hidden
        self.w1, self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _enconde_labels(self, y, k):
        # one hot vector encoding
        onehot = np.zeros((k, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.0
        return onehot

    def _initialize_weights(self):
        # connects the input layers' weights with the hidden layer
        w1 = np.random.uniform(-1, 1, size=self.hidden * (self.n_features + 1))  # gives all the weights for all connections and layers with the bias
        w1 = w1.reshape(self.hidden, self.n_features + 1)  # reshapes the weights the weights into a matrix hidden * features that is equal to the random values

        # connects the hidden layer with the output layer with the bias
        w2 = np.random.uniform(-1, 1, size=self.n_output * (self.hidden + 1))
        w2 = w2.reshape(self.n_output, self.hidden + 1)

        return w1, w2

    def _sigmoid(self, z):
        # 1 / 1 + np.exp(-z)
        return 1 / (1 + (np.exp(-z)))

    def _sigmoid_gradient(self, z):
        # error correction using the sigmoid function
        sg = self._sigmoid(z)
        return sg * (1 - sg)

    def _add_bias_unit(self, X, how='column'):
        if how == 'column':
            X_new = np.ones((X.shape[0], X.shape[1] + 1))
            X_new[:, 1:] = X
        elif how == 'row':
            X_new = np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
        else:
            raise AttributeError("`how` must be `column` or `row`")

        return X_new

    def _feedfoward(self, X, w1, w2):
        # feeds foward the net_input to the other neuron
        # applies activation

        a1 = self._add_bias_unit(X, how='column')  # bias for hidden layer and input layer
        z2 = w1.dot(a1.T)  # hidden layer with weight * bias unit  coming from the input layer meaning (X)
        a2 = self._sigmoid(z2)  # applies activation function to the result from each neuron in the hidden layer
        a2 = self._add_bias_unit(a2, how='row')   # adds bias unit to each neuron in the hidden layer
        z3 = w2.dot(a2)  # hidden layer with weight * bias unit   coming from the hidden layer to the output layer  meaning (hidden layer 1)
        a3 = self._sigmoid(z3)  # applies activation function from hidden to output layer
        return a1, z2, a2, z3, a3

    def _l2_reg(self, lambda_, w1, w2):
        # regularize on least squares with the sum squared of the weight error so basically the error or loss + reg
        # does it for each of the layers input hidden, hidden output

        return (lambda_ / 2) * (np.sum(w1[:, 1:] ** 2) + np.sum(w2[:, 1:] ** 2))  # lambda represents the regularization strength

    def _l1_reg(self, lambda_, w1, w2):
        # regularize on least squares
        # regularizes with the sum of the absolute value of the weight error or loss
        # lambda still represents strength of penalization
        return (lambda_ / 2) * (np.abs(w1[:, 1:]).sum() + np.abs(w2[:, 1:]).sum())

    def _get_cost(self, y_enc, output, w1, w2):
        # y_enc represents the y_encoded labels
        # based on the logistic cost function
        term1 = -y_enc * (np.log(output))  # error
        term2 = (1 - y_enc) * np.log(1 - output)
        cost = np.sum(term1 - term2)
        l1_term = self._l1_reg(self.l1, w1, w2)
        l2_term = self._l2_reg(self.l2, w1, w2)
        cost = cost + l1_term + l2_term
        return cost

    def _get_gradient(self, a1, a2, a3, z2, y_enc, w1, w2):
        # backpropagation

        sigma3 = a3 - y_enc  # gets the error from the last layer (output - the actual label)
        z2 = self._add_bias_unit(z2, how='row')
        sigma2 = w2.T.dot(sigma3) * self._sigmoid_gradient(z2)  # error for weights in hidden layer to output layer applied to weights by the sigma3
        sigma2 = sigma2[1:, :]
        grad1 = sigma2.dot(a1)  # derivation of cost function so by how much we correct from output to hidden, weight updates
        grad2 = sigma3.dot(a2.T)  # derivation of cost so by how much we correct from hidden to input , weight updates

        # regularize weight updates
        grad1[:, 1:] += (w1[:, 1:] * (self.l1 + self.l2))  # applies regulatirazion from input to hidden layers
        grad2[:, 1:] += (w2[:, 1:] * (self.l1 + self.l2))  # applies regularization from hidden to output

        return grad1, grad2

    def predict(self, X):
        # returns the prediction for the X features
        # example:
        # np.argmax([-0.00138,-6.58]) returns 0
        # np.argmax([-5.00138,-0.73]) returns 1
        # in this, it will give the argmax of the n_output variables!!!!!! 0-9 for example
        # thus the predicted labels
        a1, z2, a2, z3, a3 = self._feedfoward(X, self.w1, self.w2)
        y_pred = np.argmax(z3, axis=0)  # imagine an array with 0 and 1, np.argmax(returns the max in that row) {0: 0.2323,1: 0.982} will return a 1 becuase it is the max
        return y_pred

    def fit(self, X, y, print_progress=False):
        self.cost_ = []
        X_data, y_data = X.copy(), y.copy()
        y_enc = self._enconde_labels(y, self.n_output)  # encode according the the number of output layers

        delta_w1_prev = np.zeros(self.w1.shape)  # change in gradient
        delta_w2_prev = np.zeros(self.w2.shape)  # change in gradient

        for i in range(self.epochs):  # number of steps, each step trains one batch

            # adaptive learning rate
            self.eta /= (1 + self.decrease_const * 1)

            if print_progress:
                sys.stderr.write('\rEpoch {}/{}'.format(i + 1, self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                X_data, y_enc = X_data.iloc[idx], y_enc[:, idx]

            mini = np.array_split(range(y_data.shape[0]), self.minibatches)

            for idx in mini:

                # feedfoward

                a1, z2, a2, z3, a3 = self._feedfoward(X_data.iloc[idx], self.w1, self.w2)

                cost = self._get_cost(y_enc=y_enc[:, idx], output=a3, w1=self.w1, w2=self.w2)

                self.cost_.append(cost)

                # compute gradient via backpropagation

                grad1, grad2 = self._get_gradient(a1=a1, a2=a2, a3=a3, z2=z2, y_enc=y_enc[:, idx], w1=self.w1, w2=self.w2)

                # update weights

                delta_w1, delta_w2 = self.eta * grad1, self.eta * grad2  # weight update
                self.w1 -= (delta_w1 + (self.alpha * delta_w1_prev))  # adds a factor to the previous weight for faster learning
                self.w2 -= (delta_w2 + (self.alpha * delta_w2_prev))  # does the w - gradient(which minimizes the cost function, and optimizes W)

                delta_w1_prev, delta_w2_prev = delta_w1, delta_w2

        return self
