"""
matsjfunke
"""

import matplotlib.pyplot as plt
import numpy as np
from cifar_10_utils import load_cifar10


def rgb2gray_weighted(images):
    """
    Convert RGB images to grayscale using weighted average (luminance).
    param images: np.array, shape (num_images, height, width, 3)
    return: np.array, shape (num_images, height, width)
    """
    return np.dot(images, [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    # Convert to grayscale using weighted average
    train_images_gray_weighted = rgb2gray_weighted(train_images)
    test_images_gray_weighted = rgb2gray_weighted(test_images)

    print("Shape of the first grayscale image (weighted average):", train_images_gray_weighted[0].shape)

    plt.figure(figsize=(10, 4))
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(train_images[0])
    plt.title("Original Image")

    # Grayscale image
    plt.subplot(1, 2, 2)
    plt.imshow(train_images_gray_weighted[0], cmap="gray")
    plt.title("Grayscale Image (Weighted Average)")

    plt.show()
