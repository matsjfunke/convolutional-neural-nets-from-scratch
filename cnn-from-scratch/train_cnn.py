"""
matsjfunke
"""

import numpy as np
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from scipy.signal import convolve2d


def forward_pass(input_img_array, num_kernels, kernel_size):
    # initalize kernels with random numbers
    kernels = [np.random.randn(kernel_size, kernel_size) for _ in range(num_kernels)]

    # convolve image with kernels to create feature_maps
    feature_maps = []
    for kernel in kernels:
        feature_map = convolve2d(input_img_array, kernel, mode="valid")
        feature_maps.append(feature_map)

    return feature_maps


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    # Convert to grayscale using weighted average
    train_images_gray = rgb2gray_weighted(train_images)
    test_images_gray = rgb2gray_weighted(test_images)

    output = forward_pass(train_images_gray[0], 2, 3)
    print(output)