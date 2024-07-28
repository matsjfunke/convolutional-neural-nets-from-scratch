"""
matsjfunke
"""

import numpy as np
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from neural_net_utils import relu
from scipy.signal import convolve2d


def forward_pass(input_img_array, num_kernels, kernel_size):
    # initalize kernels with random numbers
    kernels = [np.random.randn(kernel_size, kernel_size) for _ in range(num_kernels)]

    # convolve image with kernels to create feature_maps
    feature_maps = []
    for kernel in kernels:
        feature_map = convolve2d(input_img_array, kernel, mode="valid")
        # apply relu for non-linearity
        feature_map = relu(feature_map)
        feature_maps.append(feature_map)

    # Stack feature maps into a 3D tensor
    output_tensor = np.stack(feature_maps, axis=-1)

    return output_tensor


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    # Convert to grayscale using weighted average
    train_images_gray = rgb2gray_weighted(train_images)
    test_images_gray = rgb2gray_weighted(test_images)

    output = forward_pass(train_images_gray[0], 2, 3)
    print(output.shape)
    # output tensor shape of (30, 30, 2) --> the input image has been convolved with 2 kernels, resulting in two feature maps of size 30x30
    # output_tensor shape (28, 28, 5) indicates 5 kernels of size 5x5
