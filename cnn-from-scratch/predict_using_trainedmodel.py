"""
matsjfunke
"""

import pickle

import numpy as np

# Import your custom utilities (assuming they are in the same directory or properly referenced)
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from neural_net_utils import calc_conv_output_size, cross_entropy_loss_gradient, init_weights_biases, max_pooling, relu, relu_derivative, softmax
from scipy.signal import convolve2d


class NeuralNetwork:
    def __init__(self, input_shape, num_kernels, kernel_size, pooling_kernel_size, stride, hidden_layer_sizes, num_classes):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.pooling_kernel_size = pooling_kernel_size
        self.stride = stride
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_classes = num_classes

        # Initialize kernels for convolutional layer
        self.kernels = [np.random.randn(kernel_size, kernel_size) for _ in range(num_kernels)]
        self.kernels_grads = [np.zeros_like(kernel) for kernel in self.kernels]

        # Initialize weights and biases for hidden layers
        self.hidden_layers = []
        prev_size = calc_conv_output_size(input_shape, kernel_size, pooling_kernel_size, stride, num_kernels)
        for layer_size in hidden_layer_sizes:
            weights, biases = init_weights_biases(num_inputs=prev_size, num_outputs=layer_size)
            self.hidden_layers.append((weights, biases))
            prev_size = layer_size

        # Initialize weights and biases for output layer
        self.output_weights, self.output_biases = init_weights_biases(num_inputs=prev_size, num_outputs=num_classes)

    def save_params(self, filename):
        with open(filename, "wb") as f:
            pickle.dump({"kernels": self.kernels, "hidden_layers": self.hidden_layers, "output_weights": self.output_weights, "output_biases": self.output_biases}, f)

    def load_params(self, filename):
        with open(filename, "rb") as f:
            params = pickle.load(f)
            self.kernels = params["kernels"]
            self.hidden_layers = params["hidden_layers"]
            self.output_weights = params["output_weights"]
            self.output_biases = params["output_biases"]

    def convolutional_layer(self, input_img_array):
        feature_maps = []
        for kernel in self.kernels:
            feature_map = convolve2d(input_img_array, kernel, mode="valid")
            feature_map = relu(feature_map)
            feature_map = max_pooling(feature_map, self.pooling_kernel_size, self.stride)
            feature_maps.append(feature_map)

        output_tensor = np.stack(feature_maps, axis=-1)
        flattened_output = output_tensor.flatten()
        return flattened_output

    def hidden_layer(self, input, weights, biases):
        return relu(np.dot(input, weights) + biases)

    def softmax_output_layer(self, hidden_layer_output):
        return softmax(np.dot(hidden_layer_output, self.output_weights) + self.output_biases)

    def forward_pass(self, input_img):
        conv_output = self.convolutional_layer(input_img)

        hidden_outputs = []
        hidden_output = conv_output
        for weights, biases in self.hidden_layers:
            hidden_output = self.hidden_layer(hidden_output, weights, biases)
            hidden_outputs.append(hidden_output)

        probabilities = self.softmax_output_layer(hidden_output)
        return probabilities, conv_output, hidden_outputs


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    train_images_gray = rgb2gray_weighted(train_images)
    test_images_gray = rgb2gray_weighted(test_images)

    nn = NeuralNetwork(
        input_shape=test_images_gray[0].shape, num_kernels=2, kernel_size=3, pooling_kernel_size=3, stride=2, hidden_layer_sizes=[128, 64], num_classes=len(label_names)
    )

    # Load the trained parameters
    nn.load_params("trained_model.pkl")

    # Example prediction
    import random

    pred_index = random.randint(0, 9999)
    probabilities, _, _ = nn.forward_pass(test_images_gray[pred_index])
    predicted_label = np.argmax(probabilities)
    print(f"Predicted index: {pred_index}, predicted label: {label_names[predicted_label]}, actual label: {label_names[test_labels[pred_index]]}")
    print(f"Predicted probabilities: {probabilities}")
