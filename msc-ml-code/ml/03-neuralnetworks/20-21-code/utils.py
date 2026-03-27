import numpy as np


def sigmoid(Z):
    '''
    The sigmoid function takes in real numbers in any range and
    squashes it to a real-valued output between 0 and 1.
    '''
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_derivative(A_next):
    '''
    The sigmoid derivative function is
	.. math::
		 \sigma(x) \star (1-\sigma(x)).
    A_next is just \sigma(x)
    '''
    g_prime = A_next * (1-A_next)
    return g_prime
