"""
matsjfunke
"""

import numpy as np


def max_pooling(feature_map, pooling_kernel_size=2, stride=2):
    # Calculate the dimensions of the pooled feature map
    pooled_height = (feature_map.shape[0] - pooling_kernel_size) // stride + 1
    pooled_width = (feature_map.shape[1] - pooling_kernel_size) // stride + 1

    # Initialize the pooled feature map
    pooled_feature_map = np.zeros((pooled_height, pooled_width))

    # Extract pooling windows and perform max pooling
    for i in range(pooled_height):
        for j in range(pooled_width):
            start_i = i * stride
            start_j = j * stride
            pooled_feature_map[i, j] = np.max(feature_map[start_i : start_i + pooling_kernel_size, start_j : start_j + pooling_kernel_size])

    return pooled_feature_map


def calc_conv_output_size(input_size, kernel_size, pooling_kernel_size, stride, num_kernels):
    # Feature map size after convolution
    conv_height = input_size[0] - kernel_size + 1
    conv_width = input_size[1] - kernel_size + 1

    # Feature map size after pooling
    pooled_height = (conv_height - pooling_kernel_size) // stride + 1
    pooled_width = (conv_width - pooling_kernel_size) // stride + 1

    return pooled_height * pooled_width * num_kernels


def init_weights_biases(num_inputs, num_outputs):
    weights = np.random.randn(num_inputs, num_outputs) * 0.01
    biases = np.zeros(num_outputs)
    return weights, biases


def relu(feature_map):
    return np.maximum(0, feature_map)


def relu_derivative(x):
    return (x > 0).astype(int)


def softmax(logits):
    """
    Parameters: logits (raw scores) from the output layer
    Returns: computed probabilities for each class
    """
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities


def cross_entropy_loss_gradient(true_labels, predicted_probs):
    """
    Parameters:
    - np.ndarray One-hot encoded true labels.
    - np.ndarray Predicted probabilities from the network.

    Returns:
    - float: Cross-entropy loss.
    - np.ndarray: Gradient of the loss with respect to the predicted probabilities.
    """
    loss = -np.sum(true_labels * np.log(predicted_probs))
    grad = predicted_probs - true_labels
    return loss, grad


