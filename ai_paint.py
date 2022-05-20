import mnist
import pygame
import numpy as np
from time import time
from tools import sigmoid, generate_weights
pygame.init()

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
            err = delta.dot(weights[idx].T)
            weights[idx] += neurons[idx].T.dot(delta) * EPSILON


# проверяем результат
def predict(x):
    y = x
    for w in weights:
        y = sigmoid(np.dot(y, w))
    return y


print(f'TRAINED BY {time() - start_time} SEC')

win = pygame.display.set_mode((700, 700))
grid = [[0 for col in range(28)] for row in range(28)]
is_painting = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            is_painting = True
        elif event.type == pygame.MOUSEBUTTONUP:
            is_painting = False
        elif event.type == pygame.KEYDOWN:
            if event.unicode == 'p':
                x = []
                for row in grid:
                    x.extend(row)
                y = predict(x)

                print(y.tolist().index(max(y)))
            elif event.unicode == 'c':
                grid = [[0 for col in range(28)] for row in range(28)]

    mouse_x, mouse_y = pygame.mouse.get_pos()

    for y, row in enumerate(grid):
        for x, col in enumerate(row):
            color = 'red'
            if not col:
                color = 'white'

            rect = pygame.draw.rect(win, color, (x * 25, y * 25, 25, 25))

            if rect.collidepoint(mouse_x, mouse_y) and is_painting:
                grid[y][x] = 1

    pygame.display.update()
