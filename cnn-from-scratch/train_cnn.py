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

    def back_prop(self, probabilities, hidden_output, true_label, learning_rate=0.001):
        # Compute the loss and its gradient
        loss, loss_grad = cross_entropy_loss_gradient(true_label, probabilities)
        print(f"Cross-Entropy Loss: {loss}")
        print(f"Loss Gradient: {loss_grad}")

        print("pre output_weight", self.output_weights[0])

        # gradient descent on output layer
        output_weights_grad = np.outer(hidden_output, loss_grad)
        output_biases_grad = loss_grad
        self.output_weights -= learning_rate * output_weights_grad
        self.output_biases -= learning_rate * output_biases_grad

        print("post output_weight", self.output_weights[0])

        # TODO: Implement the gradient computation for hidden layers and update step for weights and biases

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
    probabilities, hidden_output = nn.forward_pass(train_images_gray[0])
    print(f"The predicted class is: {label_names[np.argmax(probabilities)]}, actual class is: {label_names[train_labels[0]]}")

    nn.back_prop(probabilities, hidden_output, true_label=train_labels[0], learning_rate=0.01)

    # TODO: Implement the gradient computation for convolutiona layer and update step for weights and biases
    # Backpropagate through Flatten Layer: Reshape the gradient to match the dimensions of the pooled output.
    # Backpropagate through Pooling Layer: Use the appropriate method to backpropagate through the pooling layer. For max pooling, this involves routing the gradients back to the positions of the maximum values.
    # Backpropagate through Convolutional Layer: Apply the ReLU derivative to the gradients of the convolutional layer's output. Compute gradients for the convolutional filters. Compute the gradient with respect to the input image (this can be used for further upstream layers if needed).
