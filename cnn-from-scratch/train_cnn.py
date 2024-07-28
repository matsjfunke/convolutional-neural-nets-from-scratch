"""
matsjfunke
"""

import numpy as np
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from neural_net_utils import calc_conv_output_size, cross_entropy_loss_gradient, init_weights_biases, max_pooling, relu, softmax
from scipy.signal import convolve2d


class NeuralNetwork:
    def __init__(self, input_shape, num_kernels, kernel_size, pooling_kernel_size, stride, hidden_layer_sizes, num_classes):
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.num_classes = num_classes

        # Initialize kernels for convolutional layer
        self.kernels = [np.random.randn(kernel_size, kernel_size) for _ in range(num_kernels)]

        # Initialize weights and biases for hidden layers
        self.hidden_layers = []
        prev_size = calc_conv_output_size(input_shape, kernel_size, pooling_kernel_size, stride, num_kernels)
        for layer_size in hidden_layer_sizes:
            weights, biases = init_weights_biases(num_inputs=prev_size, num_outputs=layer_size)
            self.hidden_layers.append((weights, biases))
            prev_size = layer_size

        # Initialize weights and biases for output layer
        self.output_weights, self.output_biases = init_weights_biases(num_inputs=prev_size, num_outputs=num_classes)

    def convolutional_layer(self, input_img_array):
        # Convolve image with kernels to create feature_maps
        feature_maps = []
        for kernel in self.kernels:
            feature_map = convolve2d(input_img_array, kernel, mode="valid")
            # Apply relu for non-linearity
            feature_map = relu(feature_map)
            # Apply pooling (reducing dimensions of feature maps) to decrease computational complexity and retaining essential features.
            feature_map = max_pooling(feature_map, pooling_kernel_size=3, stride=2)
            feature_maps.append(feature_map)

        # Stack feature maps into a 3D tensor
        output_tensor = np.stack(feature_maps, axis=-1)
        # Flatten the output tensor to 1D
        flattened_output = output_tensor.flatten()
        return flattened_output

    def hidden_layer(self, input, weights, biases):
        # Compute layer output (logits) & apply ReLU
        return relu(np.dot(input, weights) + biases)

    def softmax_output_layer(self, hidden_layer_output):
        # Compute layer output (logits) & apply softmax
        return softmax(np.dot(hidden_layer_output, self.output_weights) + self.output_biases)

    def forward_pass(self, input_img):
        conv_output = self.convolutional_layer(input_img)

        hidden_output = conv_output
        for weights, biases in self.hidden_layers:
            hidden_output = self.hidden_layer(hidden_output, weights, biases)

        probabilities = self.softmax_output_layer(hidden_output)
        return probabilities, hidden_output


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    # Convert to grayscale using weighted average
    train_images_gray = rgb2gray_weighted(train_images)
    test_images_gray = rgb2gray_weighted(test_images)

    # Initialize the neural network
    nn = NeuralNetwork(
        input_shape=train_images_gray[0].shape, num_kernels=2, kernel_size=3, pooling_kernel_size=3, stride=2, hidden_layer_sizes=[128, 64], num_classes=len(label_names)
    )

    # Perform forward pass
    probabilities, _ = nn.forward_pass(train_images_gray[0])
    print(f"The predicted class is: {label_names[np.argmax(probabilities)]}, actual class is: {label_names[train_labels[0]]}")

    # back_prop(probabilities)
    # def back_prop(probabilities):
    # true_label = np.zeros(len(label_names))
    # true_label[train_labels[0]] = 1

    # # Compute the loss and its gradient
    # loss, loss_grad = cross_entropy_loss_gradient(true_label, probabilities)
    # print(f"actual label: {label_names[train_labels[0]]}")
    # print(f"Cross-Entropy Loss: {loss}")
    # print(f"Loss Gradient: {loss_grad}")
