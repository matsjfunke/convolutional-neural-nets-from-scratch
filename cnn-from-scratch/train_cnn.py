"""
matsjfunke
"""

import numpy as np
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from neural_net_utils import max_pooling, relu
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
        # apply pooling (reducing dimensions of feature maps) to decrease computational complexity and retaining essential features.
        feature_map = max_pooling(feature_map, kernel_size=3, stride=2)
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

    output = forward_pass(train_images_gray[0], num_kernels=2, kernel_size=3)
    print(output.shape)
