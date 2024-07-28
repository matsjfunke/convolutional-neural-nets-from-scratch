"""
matsjfunke
"""

import numpy as np


def relu(feature_map):
    return np.maximum(0, feature_map)


def softmax(logits):
    """
    Parameters: logits (raw scores) from the output layer
    Returns: computed probabilities for each class
    """
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities


def init_weights_biases(num_inputs, num_outputs):
    weights = np.random.randn(num_inputs, num_outputs) * 0.01
    biases = np.zeros(num_outputs)
    return weights, biases


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
