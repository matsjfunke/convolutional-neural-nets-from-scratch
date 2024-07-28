"""
matsjfunke
"""

import numpy as np
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from neural_net_utils import init_weights_biases, max_pooling, relu, softmax
from scipy.signal import convolve2d


def convolutional_layer(input_img_array, num_kernels, kernel_size):
    # Initalize kernels with random numbers
    kernels = [np.random.randn(kernel_size, kernel_size) for _ in range(num_kernels)]

    # Convolve image with kernels to create feature_maps
    feature_maps = []
    for kernel in kernels:
        feature_map = convolve2d(input_img_array, kernel, mode="valid")
        # Apply relu for non-linearity
        feature_map = relu(feature_map)
        # Apply pooling (reducing dimensions of feature maps) to decrease computational complexity and retaining essential features.
        feature_map = max_pooling(feature_map, kernel_size=3, stride=2)
        feature_maps.append(feature_map)

    # Stack feature maps into a 3D tensor
    output_tensor = np.stack(feature_maps, axis=-1)

    # Flatten the output tensor to 1D
    flattened_output = output_tensor.flatten()
    return flattened_output


def hidden_layer(flattened_input, output_size):
    input_size = flattened_input.shape[0]
    weights, biases = init_weights_biases(num_inputs=input_size, num_outputs=output_size)

    # Compute layer output (logits) & apply ReLU
    activated_output = relu(np.dot(flattened_input, weights) + biases)
    return activated_output


def softmax_output_layer(hidden_layer_output, num_classes):
    input_size = hidden_layer_output.shape[0]
    weights, biases = init_weights_biases(num_inputs=input_size, num_outputs=num_classes)

    # Compute layer output (logits) & apply softmax
    activated_output = softmax(np.dot(hidden_layer_output, weights) + biases)
    return activated_output


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    # Convert to grayscale using weighted average
    train_images_gray = rgb2gray_weighted(train_images)
    test_images_gray = rgb2gray_weighted(test_images)

    conv_output = convolutional_layer(train_images_gray[0], num_kernels=2, kernel_size=3)

    # Pass the flattened output through the fully connected layer
    output_hidden_1 = hidden_layer(conv_output, output_size=128)
    output_hidden_2 = hidden_layer(output_hidden_1, output_size=8)

    probabilities = softmax_output_layer(output_hidden_2, len(label_names))
    print(np.sum(probabilities))
    # Determine the index of the highest probability
    predicted_index = np.argmax(probabilities)

    # Get the corresponding class label
    predicted_class = label_names[predicted_index]

    print(f"The predicted class is: {predicted_class}")
    print(f"The predicted class probability is: {probabilities[predicted_index]:.6f}")
