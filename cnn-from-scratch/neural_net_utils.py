"""
matsjfunke
"""

import numpy as np


def relu(feature_map):
    return np.maximum(0, feature_map)

def max_pooling(feature_map, kernel_size=2, stride=2):
    # Calculate the dimensions of the pooled feature map
    pooled_height = (feature_map.shape[0] - kernel_size) // stride + 1
    pooled_width = (feature_map.shape[1] - kernel_size) // stride + 1

    # Initialize the pooled feature map
    pooled_feature_map = np.zeros((pooled_height, pooled_width))

    # Extract pooling windows and perform max pooling
    for i in range(pooled_height):
        for j in range(pooled_width):
            start_i = i * stride
            start_j = j * stride
            pooled_feature_map[i, j] = np.max(feature_map[start_i : start_i + kernel_size, start_j : start_j + kernel_size])

    return pooled_feature_map



