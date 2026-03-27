import numpy as np
from matplotlib import pyplot as plt

from utils import sigmoid, sigmoid_derivative


class NeuralNet():

    def __init__(self, layers=[13, 8, 1], learning_rate=0.001, iterations=100, lmd=0):
        self.w = {}
        self.b = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lmd = lmd
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None
        self.A = {}
        self.Z = {}
        self.dW = {}
        self.dB = {}
        self.dZ = {}
        self.dA = {}

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(42)  # Seed the random number generator
        L = len(self.layers)
        for l in range(1, L):
            self.w[l] = np.random.randn(self.layers[l], self.layers[l-1])
            self.b[l] = np.random.randn(self.layers[l], 1)

    def compute_cost_regression(self, AL):
        m = self.y.shape[0]
        layers = len(AL) // 2
        Y_pred = AL['A' + str(layers)]
        cost = (1 / (2 * m)) * np.sum(np.square(Y_pred - self.y.T))
        reg_sum = 0
        for l in range(1, len(self.layers)):
            reg_sum += (np.sum(np.square(self.w[l])))
        L2_regularization_cost = reg_sum * (self.lmd / (2 * m))
        return cost + L2_regularization_cost

    def compute_cost_regression_derivative(self, AL):
        return AL - self.y.T

    def update_params(self, grads):
        layers = len(self.w)
        params_updated = {}
        for i in range(1, layers + 1):
            self.w[i] = self.w[i] - self.learning_rate * grads['W' + str(i)]
            self.b[i] = self.b[i] - self.learning_rate * grads['B' + str(i)]
        return params_updated

    def backpropagation_regression(self, values):
        layers = len(self.w)
        m = len(self.y)
        grads = {}
        for i in range(layers, 0, -1):
            if i == layers:
                dA = 1 / m * self.compute_cost_regression_derivative(values['A' + str(i)])
                dZ = dA
            else:
                dA = np.dot(self.w[i + 1].T, dZ)
                dZ = np.multiply(dA, sigmoid_derivative(values['A' + str(i)]))
            if i == 1:
                grads['W' + str(i)] = 1 / m * np.dot(dZ, self.X) + self.lmd * self.w[i]
                grads['B' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            else:
                grads['W' + str(i)] = 1 / m * np.dot(dZ, values['A' + str(i - 1)].T) + self.lmd * self.w[i]
                grads['B' + str(i)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        return grads

    def forward_propagation_regression(self):
        '''
        Performs the forward propagation
        '''
        layers = len(self.w)
        values = {}
        for i in range(1, layers + 1):
            if i == 1:
                values['Z' + str(i)] = np.dot(self.w[i], self.X.T) + self.b[i]
                values['A' + str(i)] = sigmoid(values['Z' + str(i)])
            else:
                values['Z' + str(i)] = np.dot(self.w[i], values['A' + str(i - 1)]) + self.b[i]
                if i == layers:
                    values['A' + str(i)] = values['Z' + str(i)]
                else:
                    values['A' + str(i)] = sigmoid(values['Z' + str(i)])
        return values

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()  # initialize weights and bias

        for i in range(self.iterations):
            A_list = self.forward_propagation_regression()
            cost = self.compute_cost_regression(A_list)
            grads = self.backpropagation_regression(A_list)
            self.update_params(grads)
            self.loss.append(cost)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        self.X = X
        AL = self.forward_propagation_regression()
        layers = len(AL) // 2
        Y_pred = AL['A' + str(layers)]
        return Y_pred

    def rmse(self, y, yhat):
        '''
        Calculates the error between the predicted valuea and the truth labels
        '''
        m = y.shape[0]
        error = yhat - y.T
        rmse = 1/m * np.sqrt(error @ error.T)
        return rmse

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("squared error")
        plt.title("Loss curve for training")
        plt.show()
