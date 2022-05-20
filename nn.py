import numpy as np
from tools import sigmoid, generate_weights, show_convergence

# объявляем переменные
X = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1]
])
Y = np.array([0, 0, 1, 1])
inputs_num = 3
outputs_num = 1
hidden_layers = [10]
EPSILON = 0.1
EPOCH_NUM = 10000
BATCH_SIZE = 1
epoch_list = []
error_list = []

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

# тренируем веса
for epoch in range(EPOCH_NUM):
    for batch in range(len(X) // BATCH_SIZE):
        x = X[BATCH_SIZE * batch:BATCH_SIZE * (batch + 1)]
        y = Y[BATCH_SIZE * batch:BATCH_SIZE * (batch + 1)]

        neurons = [x]

        for idx, weight in enumerate(weights):
            neurons.append(sigmoid(np.dot(neurons[-1], weight)))

        predict_y = neurons[-1]
        err = y - predict_y

        for idx in range(len(weights) - 1, -1, -1):
            """
            neurons[idx + 1] - output
            neurons[idx] - input
            """

            delta = err * sigmoid(neurons[idx + 1], True)
            grad = neurons[idx].T.dot(delta)
            weights[idx] += grad * EPSILON

            error_list.append(err)
            epoch_list.append(epoch)

            err = delta.dot(weights[idx].T)


# проверяем результат
def predict(x):
    y = x
    for w in weights:
        y = sigmoid(np.dot(y, w))
    return y


print(predict(np.array([1, 0, 1])))
print(predict(np.array([0, 0, 1])))

show_convergence(epoch_list, error_list)
