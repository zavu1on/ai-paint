import time
import numpy as np
import mnist
from neural_network import NeuralNetwork


X = mnist.train_images()
Y = mnist.train_labels()
train_X = []
train_Y = np.array([[0] * y + [1] + [0] * (9-y) for y in Y])

for i in X:
    train_X.append([])

    for j in i:
        train_X[-1].extend([int(bool(el)) for el in j])

print('TRAINING...')

t = time.time()

nn = NeuralNetwork(784, [800, 10])
nn.train(np.array(train_X), np.array(train_Y), 20, 250, 0.01)

print(f'TRAINED BY {time.time() - t} SEC')

X = mnist.test_images()[:100]
Y = mnist.test_labels()[:100]
test_X = []
test_Y = np.array([[0] * y + [1] + [0] * (9 - y) for y in Y])
for i in X:
    test_X.append([])

    for j in i:
        test_X[-1].extend([int(bool(el)) for el in j])

for x, y in zip(test_X, test_Y):
    prediction = nn.predict(x)

    print(
        prediction.tolist().index(max(prediction)),
        y.tolist().index(max(y)),
    )
