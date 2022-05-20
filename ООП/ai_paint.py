import time
import numpy as np
import pygame
import mnist
from neural_network import NeuralNetwork

pygame.init()

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
                y = nn.predict(x)

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
