"""
matsjfunke
"""

import numpy as np


def init_weights_biases(num_inputs, num_outputs):
    stddev = np.sqrt(2.0 / num_inputs)
    weights = np.random.randn(num_inputs, num_outputs) * stddev
    biases = np.zeros(num_outputs)
    return weights, biases


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


def convolve2d(img, kernel):
    """
    Perform a 2D convolution operation between the input image and the kernel manually.
    Mode: 'valid' (no padding, output size is reduced).
    """
    # Flip the kernel both horizontally and vertically
    kernel = np.flipud(np.fliplr(kernel))

    # Get the dimensions of the input image and kernel
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the dimensions of the output feature map
    output_height = img_height - kernel_height + 1
    output_width = img_width - kernel_width + 1

    # Initialize the output feature map with zeros
    output = np.zeros((output_height, output_width))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current region of the image
            region = img[i : i + kernel_height, j : j + kernel_width]
            # Apply the kernel to the current region (element-wise multiplication and sum)
            output[i, j] = np.sum(region * kernel)

    return output


def relu(x):
    return np.clip(np.maximum(0, x), 0, 1e10)  # Clip to avoid extreme values


def relu_derivative(x):
    return (x > 0).astype(int)


def softmax(logits):
    """
    Parameters: logits (raw scores) from the output layer
    Returns: computed probabilities for each class
    """
    # Check for NaNs or Infs in logits
    if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
        raise ValueError("Logits contain NaNs or Infs")

    # Subtract the max for numerical stability
    logits = np.asarray(logits)  # Ensure logits is an array
    max_logits = np.max(logits)
    stable_logits = logits - max_logits

    # Clip logits to avoid overflow in exponentiation
    clipped_logits = np.clip(stable_logits, -700, 700)

    exp_logits = np.exp(clipped_logits)
    return exp_logits / np.sum(exp_logits)


def cross_entropy_loss_gradient(true_labels, predicted_probs, epsilon=1e-15):
    """
    Parameters:
    - np.ndarray One-hot encoded true labels.
    - np.ndarray Predicted probabilities from the network.

    Returns:
    - float: Cross-entropy loss.
    - np.ndarray: Gradient of the loss with respect to the predicted probabilities.
    """
    # Clip probabilities to avoid log(0)
    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    loss = -np.sum(true_labels * np.log(predicted_probs))
    gradient = predicted_probs - true_labels
    return loss, gradient
