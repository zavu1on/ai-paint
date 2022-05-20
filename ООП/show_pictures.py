import pygame
import mnist
pygame.init()

X = mnist.train_images()
idx = 0

win = pygame.display.set_mode((700, 700))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                idx += 1
            elif event.key == pygame.K_LEFT:
                idx -= 1

    win.fill('#ffffff')

    for y, row in enumerate(X[idx]):
        for x, col in enumerate(row):
            color = 'red'
            if not col:
                color = 'white'

            pygame.draw.rect(win, color, (x * 25, y * 25, 25, 25))

    pygame.display.update()
