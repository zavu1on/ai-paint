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
# code ...

start_time = time()
print('TRAINING')

# тренируем веса
# code ...

# проверяем результат
# code ...

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

# code ...
