import numpy as np
from tools import sigmoid


class NeuralNetwork:
    def __init__(self, inputs_num=1, layers: list[int] = [1]):
        self.layers = [inputs_num] + layers
        self.weights = [2 * np.random.random([inputs_num, layers[0]]) - 1]

        for layer in range(1, len(layers)):
            self.weights.append(2 * np.random.random([layers[layer - 1], layers[layer]]) - 1)

    def predict(self, x):
        y = x
        for weight in self.weights:
            y = sigmoid(np.dot(y, weight))
        return y

    def train(self, X, Y, epochs=1, batch_size=1, epsilon=1.0):
        for epoch in range(epochs):
            for batch in range(len(X) // batch_size):
                x = X[batch_size * batch: batch_size * (batch + 1)]
                ideal_y = Y[batch_size * batch: batch_size * (batch + 1)]

                layers_outputs = [x]
                for weight in self.weights:
                    layers_outputs.append(sigmoid(np.dot(layers_outputs[-1], weight)))

                predict_y = layers_outputs[-1]
                err = ideal_y - predict_y

                for idx in range(len(self.weights) - 1, -1, -1):
                    """
                    layers_outputs[idx + 1] - output
                    layers_outputs[idx] - input
                    """

                    delta = err * sigmoid(layers_outputs[idx + 1])  # derivative
                    err = delta.dot(self.weights[idx].T)
                    self.weights[idx] += layers_outputs[idx].T.dot(delta) * epsilon


if __name__ == '__main__':
    X = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 1, 0],
                  [1, 1, 1]])
    Y = np.array([0, 0, 1, 1])
    nn = NeuralNetwork(3, [10, 1])

    nn.train(X, Y, 10000, 1, 0.01)

    print(nn.predict(np.array([1, 0, 1])))
    print(nn.predict(np.array([0, 1, 1])))
