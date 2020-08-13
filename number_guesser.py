import pygame
import time
import os
import numpy as np
from tensorflow import keras

WIN_WIDTH = 600
WIN_HEIGHT = 600
white = (255, 255, 255)
radius = 15


pygame.init()
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption('Number Guesser AI')

clock = pygame.time.Clock()


def save_image(window):
    try:
        os.remove('screenshot.jpg')
    except:
        pass

    save_file = 'screenshot.jpg'
    pygame.image.save(window, save_file)

    return


def predict_image(window):

    save_image(window)

    # Initiating the Neural Network
    model = keras.models.load_model('CONV2D.h5')

    probability_model = keras.Sequential([
        model, keras.layers.Softmax()
    ])
    # --------------------------------

    # Preprocessing the image
    image = keras.preprocessing.image.load_img(
        'screenshot.jpg',
        color_mode="grayscale",
        interpolation="nearest",
        target_size=(28, 28)
    )
    array = keras.preprocessing.image.img_to_array(image)
    array = np.array([array])/255

    array.reshape(28, 28, 1)

    # --------------------------------

    prediction = probability_model.predict(array)
    print()
    print()
    print()
    print(f'I predict that this number is {np.argmax(prediction)}')
    print()
    print()
    print()
    return


def main_loop():
    running = True

    while running:
        clock.tick(100)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    win.fill((0, 0, 0))
                elif event.key == pygame.K_p:
                    predict_image(win)

        if pygame.mouse.get_pressed()[0]:

            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.circle(win, white, mouse_pos, radius)

        pygame.display.update()


main_loop()
