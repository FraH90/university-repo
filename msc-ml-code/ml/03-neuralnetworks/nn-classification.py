import numpy as np
import matplotlib.pyplot as plt

class Layer:
    '''
    Single layer in the neural network. 
    Output size = number of neurons in the layer.
    Input size = number of neurons in the previous layer (number of features in the input data for the first layer).
    l = index that identify the layer number in the neural network structure. First layer: l=0

    z = weighted sum of the inputs (coming from the previous layer); z = W * a_prev
    a = activation of the layer (output of the layer); a = sigmoid(z)

    '''
    def __init__(self, l, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.l = l
        # If it's a first layer it's an output-only layer, set W and z to nan if l=0
        # Init weights with small random values to break symmetry, avoid zero gradients and help with convergence
        np.random.seed(42)
        self.W = np.random.randn(output_size, input_size+1) * 0.01 if self.l!=0 else np.nan
        # Initialize z, vector of the weighted sum of the inputs (plus bias)
        self.z = np.zeros((output_size, 1)) if self.l!=0 else np.nan
        # Initialize a, vector of the activations, output of the current layer and input for the next
        # dim(a) = dim(z) + 1, since we add the fake feature in order to store also the bias terms in the weight matrix W
        self.a = np.zeros((output_size + 1, 1))
        # Vectors to compute the backpropagation
        self.da = np.zeros((output_size + 1, 1))
        self.dz = np.zeros((output_size, 1))
        self.dW = np.zeros((output_size, input_size+1))
    
    def __repr__(self):
        return f'Layer #{self.l}, #neurons={self.output_size}, each neuron weighting {self.input_size} inputs'
    

class NeuralNetwork:
    '''
    Implementation of a neural network, as stack of layers

    nn_sructure: list containing the dimension (number of neurons) of each layer
    The first element is the input layer, the last one is the output layer.

    learning_rate: learning rate for the gradient descent
    '''
    def __init__(self, nn_structure, learning_rate=0.01, lmd=10):
        self.nn_structure = nn_structure
        self.learning_rate = learning_rate
        self.num_layers = len(nn_structure)
        self.output = np.zeros((self.nn_structure[-1], 1))
        self.epoch = 0
        self.loss = []
        self.lmd = lmd

        # Init the layers, and create the neural network structure
        # self.layers will be a list of "layer" objects.
        self.layers = []
        for l in range(0, len(nn_structure)):
            # Input and output dimensions of each layer; the first it's an output-only layer, there's no input, set input_size=nan
            input_dim = nn_structure[l-1] if l!=0 else np.nan
            output_dim = nn_structure[l]
            # Create all the other layers
            self.layers.append(Layer(l, input_dim, output_dim))

        # Give a representation of the neural network structure
        for layer in self.layers:
            print(layer)

    def sigmoid(self, z):
        return 1 / ( 1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forwardprop_class(self, x):
        '''
        Method to perform the forward propagation to the input, for the classification case
        Since all the layers are stored in the neural network class, and in the layers we have the a,z,da,dz,dW values,
        at the end of the forward propagation we'll have beside the value of the output neurons (self.output) also all the values of z,a of the inner layers
        Those will be used later in the backpropagation
        '''
        # sanitize input sample (make it same shape as the input layer, if it not already is)
        x = x.reshape((-1,1))
        # load the sample (vector features x) in the first layer
        # remember to concatenate the fake feature, before loading the sample in the network!
        fake_feature = np.array([1]).reshape((-1,1))
        input_sample = np.concatenate((fake_feature, x))
        self.layers[0].a[:] = input_sample

        for i in range(1, len(self.layers)):
            # The activation is the "a" of the previous layer
            activation = self.layers[i-1].a
            # Compute for the current layer the weighted sum of the input from the previous layer
            self.layers[i].z[:] = np.dot(self.layers[i].W, activation)
            # Pass z through the activation function in order to get the output from the current layer
            # Remember: this output doesn't still contain the fake feature
            a_temp = self.sigmoid(self.layers[i].z[:])
            self.layers[i].a[:] = np.concatenate((fake_feature, a_temp))

        self.output[:] = self.layers[-1].a[1:]
        print(f'Final value: {self.output}')
        return self.output

    def backprop_class(self, y_sample):
        # sanitize y
        y_sample = y_sample.reshape((-1,1))
        # Ingrediends for backpropagation:
        # - All the "activated" versions of the values in each layer (the a values; this is the number inside the neuron)
        for i in range(len(self.layers)-1, 0, -1):
            if i == len(self.layers)-1 :
                dA = self.cost_classification_derivative(self.layers[i].a)
                dZ = np.multiply(dA, self.sigmoid_derivative(self.layers[i].a)) 
            else:
                dA = np.dot(self.layers[i+1].T, dZ)
                dZ = np.multiply(dA, self.sigmoid_derivative(self.layers[i].a))
            if i == 1:
    
    def update_params(self):
        pass

    def train(self, X, y, max_iterations):
        self.X = X
        self.y = y
        self.epoch = 0

        for self.epoch in range(max_iterations):
            self.forwardprop_class()
            self.cost_class()
            self.backprop_class()
            self.update_params()
    
    def cost_class(self, y):
        '''
        Problem: to compute the cost.. we must pass all the dataset to compute J. How? We're training each time passing a single sample
        One idea can be to compute a cumulative cost. But this is stochastic gd, right?
        And if we want the samples all together, compute the prediction for ALL the samples in the dataset and then compute cost? 
        '''
        # Considering multiclass classification one vs all, this compute at first the cost relative to each class (output neuron)
        # You'll end up with a vector; summing the elements of the vector (k-summation in the slides) you get the overall cost, but relative only to that sample
        # since we're doing stochastic gd, m=1, each single sample enter per time; remember to put minus sign on the summation (-1/m...)
        cost_vector = np.multiply(y, np.log(self.output)) + np.multiply((1-self.y), np.log(1-self.output))
        cost = - np.sum(cost_vector)
        # Now compute the regularization part of the cost, by summing the square of every weight in each layer of the network (and dividing by 2m, but m=1 for now)
        for l in range(1, len(self.layers)):
            reg_sum += (np.sum(np.square(self.layers[l].W)))
        l2_reg_cost = reg_sum * (self.lmd / 2)
        cost = cost + l2_reg_cost
        self.loss.append(cost)
        return cost


def main():
    input = np.array([7,25,3])
    nn = NeuralNetwork([input.shape[0], 100, 150, 50, 1], 0.01)
    nn.forwardprop(input)

if __name__ == "__main__":
    main()


# For SGD, the input "vector" becomes the entire dataset X; but now the samples must be organized as columns
# Example first layer: when u compute theta1*x, instead of a single activation u compute all the activations relative to all the samples.
# We obtain a matrix, where each j-th column will be a "run" of a neural network on the sample j-th
# It will be like running the same neural network in parallel, for all the samples in the dataset
