import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def show_convergence(epochs, errors):
    plt.plot(epochs, errors)
    plt.show()
