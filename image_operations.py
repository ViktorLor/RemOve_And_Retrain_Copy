"""
7.01.2022
Loreth Viktor
Intepretability of Machine Learning Models
Saliency maps

Image Operations

Description:

The functions in this file are used to do different operations on images.
1. Blacking out pixels
2. Zooming in on an image
"""

import numpy as np


def black_out_random_pixels(image: np.array, pixels_ratio=0.1, seed=None):
    """
    Black out random pixels in an image.

    :param image: np.array of shape (height, width, channels)
    :param pixels_ratio: pixels to black out in relation to the total number of pixels in the image
    :return: image with blacked out pixels
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()

    # create a mask of pixels to black out
    # measure time

    mask = rng.random(image.shape)

    mask[mask > pixels_ratio] = 1
    mask[mask <= pixels_ratio] = 0

    return black_out_pixels(image, mask)


def black_out_pixels(image_: np.array, pixels_mask):
    image = image_.copy()
    image[pixels_mask == 0] = 0
    return image



def split_image_into_x_areas(image, areas=4):

    """
    Split an image into x areas.
    The number of areas must be a square number.
    returns the areas as a list of images.
    """
    if not np.sqrt(areas).is_integer():
        raise ValueError("The number of areas must be a square number.")

    singular_length = int(image.shape[0] / np.sqrt(areas))
    singular_width = int(image.shape[1] / np.sqrt(areas))
    images = []

    area_boundaries = []
    for i in range(int(np.sqrt(areas))):
        for j in range(int(np.sqrt(areas))):
            area_boundaries.append((i * singular_length, (i + 1) * singular_length, j * singular_width, (j + 1) * singular_width))
            images.append(image[i * singular_length:(i + 1) * singular_length, j * singular_width:(j + 1) * singular_width])

    return images, area_boundaries