import pickle

import numpy as np
from cifar_10_utils import load_cifar10, rgb2gray_weighted
from neural_net_utils import calc_conv_output_size, conv_weights_grad, cross_entropy_loss_gradient, init_weights_biases, max_pooling, relu, relu_derivative, softmax
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
        # Convolve image with kernels to create feature_maps
        feature_maps = []
        for kernel in self.kernels:
            feature_map = convolve2d(input_img_array, kernel, mode="valid")
            # Apply relu for non-linearity
            feature_map = relu(feature_map)
            # Apply pooling (reducing dimensions of feature maps) to decrease computational complexity and retaining essential features.
            feature_map = max_pooling(feature_map, self.pooling_kernel_size, self.stride)
            feature_maps.append(feature_map)

        # Stack feature maps into a 3D tensor
        output_tensor = np.stack(feature_maps, axis=-1)
        # Flatten the output tensor to 1D
        flattened_output = output_tensor.flatten()
        return flattened_output, output_tensor

    def hidden_layer(self, input, weights, biases):
        # Compute layer output (logits) & apply ReLU
        return relu(np.dot(input, weights) + biases)

    def softmax_output_layer(self, hidden_layer_output):
        # Compute layer output (logits) & apply softmax
        return softmax(np.dot(hidden_layer_output, self.output_weights) + self.output_biases)

    def forward_pass(self, input_img):
        conv_output, conv_pre_flatten = self.convolutional_layer(input_img)

        hidden_outputs = []
        hidden_output = conv_output
        for weights, biases in self.hidden_layers:
            hidden_output = self.hidden_layer(hidden_output, weights, biases)
            hidden_outputs.append(hidden_output)

        probabilities = self.softmax_output_layer(hidden_output)
        return probabilities, conv_output, conv_pre_flatten, hidden_outputs

    def back_prop(self, input_img, probabilities, conv_pre_flatten, conv_output, hidden_outputs, true_label, learning_rate=0.001):
        # Compute the loss and its gradient
        loss, output_loss_grad = cross_entropy_loss_gradient(true_label, probabilities)

        # Backpropagation through output layer
        hidden_output = hidden_outputs[-1]
        output_weights_grad = np.outer(hidden_output, output_loss_grad)
        output_biases_grad = output_loss_grad
        # apply gradient descent to weights & biases
        self.output_weights -= learning_rate * output_weights_grad
        self.output_biases -= learning_rate * output_biases_grad

        # Backpropagation through hidden layers (reversed)
        next_layer_grad = np.dot(output_loss_grad, self.output_weights.T) * relu_derivative(hidden_outputs[-1])
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            weights, biases = self.hidden_layers[i]
            input_to_layer = conv_output if i == 0 else hidden_outputs[i - 1]

            weights_grad = np.outer(input_to_layer, next_layer_grad)
            biases_grad = next_layer_grad

            # apply gradient descent to weights & biases
            self.hidden_layers[i] = (weights - learning_rate * weights_grad, biases - learning_rate * biases_grad)

            # Prepare next layer gradient
            if i > 0:
                next_layer_grad = np.dot(next_layer_grad, weights.T) * relu_derivative(hidden_outputs[i - 1])

        # Backpropagation through convolutional layer
        # TODO: fix this it doesnt work
        conv_output_grad = relu_derivative(conv_pre_flatten)
        kernels_grads = conv_weights_grad(input_img, self.kernels, conv_output_grad)

        # Print gradient norms
        for i, kernel in enumerate(self.kernels):
            print(f"Kernel {i} gradient norm: {np.linalg.norm(kernels_grads[i])}")

        # apply gradient descent to kernels
        for i in range(len(self.kernels)):
            self.kernels[i] -= learning_rate * kernels_grads[i]

        return loss

    def train(self, train_images, train_labels, num_epochs=10, batch_size=32, learning_rate=0.001):
        num_samples = train_images.shape[0]
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct_predictions = 0
            num_batches = num_samples // batch_size

            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_images = train_images[batch_start:batch_end]
                batch_labels = train_labels[batch_start:batch_end]

                for i in range(batch_images.shape[0]):
                    input_img = batch_images[i]
                    true_label = batch_labels[i]

                    # Forward pass
                    probabilities, conv_output, conv_pre_flatten, hidden_outputs = self.forward_pass(input_img)

                    # Compute loss and update gradients
                    loss = self.back_prop(input_img, probabilities, conv_output, conv_pre_flatten, hidden_outputs, true_label, learning_rate)

                    # Track loss and accuracy
                    epoch_loss += loss
                    if np.argmax(probabilities) == true_label:
                        correct_predictions += 1

            # Print loss and accuracy for the epoch
            average_loss = epoch_loss / num_batches
            accuracy = correct_predictions / num_samples
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, correct predictions: {correct_predictions} out of {num_samples} sampels")


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)

    train_images_gray = rgb2gray_weighted(train_images)
    test_images_gray = rgb2gray_weighted(test_images)

    nn = NeuralNetwork(
        input_shape=train_images_gray[0].shape, num_kernels=2, kernel_size=3, pooling_kernel_size=3, stride=1, hidden_layer_sizes=[128, 64], num_classes=len(label_names)
    )

    # Train the neural network
    nn.train(train_images_gray, train_labels, num_epochs=10, batch_size=32, learning_rate=0.01)

    # Save the trained parameters
    nn.save_params("trained_model.pkl")

    # Load the trained parameters (for future use)
    nn.load_params("trained_model.pkl")

    # Example prediction
    import random

    pred_index = random.randint(0, 9999)
    probabilities, _, _, _ = nn.forward_pass(test_images_gray[pred_index])
    predicted_label = np.argmax(probabilities)
    print(f"Predicted index: {pred_index}, predicted label: {label_names[predicted_label]}, actual label: {label_names[test_labels[pred_index]]}")
    print(f"Predicted probabilities: {probabilities}")
