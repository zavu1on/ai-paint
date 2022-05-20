import mnist
import numpy as np
from time import time
from tools import sigmoid, generate_weights

print('INIT VARS')

# объявляем переменные
X = mnist.train_images()
Y = mnist.train_labels()
train_X = []
train_Y = np.array([[0] * y + [1] + [0] * (9 - y) for y in Y])
for i in X:
    train_X.append([])

    for j in i:
        train_X[-1].extend([int(bool(el)) for el in j])

inputs_num = 784
outputs_num = 10
hidden_layers = [800]
EPSILON = 0.01
EPOCH_NUM = 20
BATCH_SIZE = 250

# заполняем веса
if hidden_layers:
    weights = [generate_weights([inputs_num, hidden_layers[0]])]
else:
    weights = [generate_weights([inputs_num, outputs_num])]

for layer_idx in range(1, len(layers := [*hidden_layers, outputs_num])):
    weights.append(generate_weights([
        layers[layer_idx - 1],
        layers[layer_idx],
    ]))

start_time = time()
print('TRAINING')

# тренируем веса
for epoch in range(EPOCH_NUM):
    for batch in range(len(train_X) // BATCH_SIZE):
        x = train_X[BATCH_SIZE * batch:BATCH_SIZE * (batch + 1)]
        y = train_Y[BATCH_SIZE * batch:BATCH_SIZE * (batch + 1)]

        neurons = [np.array(x)]

        for weight in weights:
            neurons.append(sigmoid(np.dot(neurons[-1], weight)))

        predict_y = neurons[-1]
        err = y - predict_y

        for idx in range(len(weights) - 1, -1, -1):
            """
            neurons[idx + 1] - output
            neurons[idx] - input
            """

            delta = err * sigmoid(neurons[idx + 1], True)
            err = delta.dot(weights[idx].T)
            weights[idx] += neurons[idx].T.dot(delta) * EPSILON


# проверяем результат
def predict(x):
    y = x
    for w in weights:
        y = sigmoid(np.dot(y, w))
    return y


print(f'TRAINED BY {time() - start_time} SEC')
print('TEST')

X = mnist.test_images()[:100]
Y = mnist.test_labels()[:100]
test_X = []
test_Y = np.array([[0] * y + [1] + [0] * (9 - y) for y in Y])
for i in X:
    test_X.append([])

    for j in i:
        test_X[-1].extend([int(bool(el)) for el in j])

for x, y in zip(test_X, test_Y):
    prediction = predict(x)

    print(
        prediction.tolist().index(max(prediction)),
        y.tolist().index(max(y)),
    )
