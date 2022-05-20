import numpy as np
from tools import sigmoid


class Perceptron:
    def __init__(self, inputs_num=1):
        self.weights = 2 * np.random.rand(inputs_num).T - 1

    def predict(self, x):
        return sigmoid(np.dot(x, self.weights))

    def train(self, X, Y, epochs=1, epsilon=1.0):
        for epoch in range(epochs):
            for j in range(len(X)):
                x = X[j]
                ideal_y = Y[j]

                predict_y = self.predict(x)

                err = (ideal_y - predict_y)
                delta = err * sigmoid(predict_y)  # derivative
                grad = np.dot(delta, x)

                self.weights += epsilon * grad


if __name__ == '__main__':
    X = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
    Y = np.array([0, 0, 1, 1])
    perceptron = Perceptron(3)

    perceptron.train(X, Y, 10000, 0.01)

    print(perceptron.predict(np.array([1, 0, 1])))
    print(perceptron.predict(np.array([0, 1, 1])))
