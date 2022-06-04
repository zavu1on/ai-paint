import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x, derivative=False):
    """ сигмоида """
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


def hyperbolic_tangent(x, derivative=False):
    """ гиперболический тангенс """
    if derivative:
        return 1 - hyperbolic_tangent(x) ** 2
    return np.tan(x ** -1)


def single_jump(x, derivative=False):
    """ единичный скачок """
    if derivative:
        return 0
    return int(x >= 0)


def show_convergence(epochs, errors):
    plt.plot(epochs, errors)
    plt.show()


def generate_weights(size):
    return 2 * np.random.random(size) - 1
