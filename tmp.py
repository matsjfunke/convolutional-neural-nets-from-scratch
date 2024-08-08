"""
matsjfunke
"""

import numpy as np


# Define the CNN layers
class Conv2D:
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / np.sqrt(filter_size * filter_size)
        self.bias = np.zeros(num_filters)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(
            (
                input.shape[0],
                self.num_filters,
                (input.shape[1] - self.filter_size + 2 * self.padding) // self.stride + 1,
                (input.shape[2] - self.filter_size + 2 * self.padding) // self.stride + 1,
            )
        )

        for i in range(self.output.shape[0]):
            for j in range(self.output.shape[1]):
                for k in range(self.output.shape[2]):
                    for l in range(self.output.shape[3]):
                        self.output[i, j, k, l] = (
                            np.sum(
                                self.input[
                                    i, :, k * self.stride : k * self.stride + self.filter_size, l * self.stride : l * self.stride + self.filter_size
                                ]
                                * self.filters[j]
                            )
                            + self.bias[j]
                        )

        return self.output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        grad_filters = np.zeros_like(self.filters)
        grad_bias = np.zeros_like(self.bias)

        for i in range(grad_output.shape[0]):
            for j in range(grad_output.shape[1]):
                for k in range(grad_output.shape[2]):
                    for l in range(grad_output.shape[3]):
                        grad_bias[j] += grad_output[i, j, k, l]
                        grad_filters[j] += (
                            grad_output[i, j, k, l]
                            * self.input[i, :, k * self.stride : k * self.stride + self.filter_size, l * self.stride : l * self.stride + self.filter_size]
                        )
                        grad_input[i, :, k * self.stride : k * self.stride + self.filter_size, l * self.stride : l * self.stride + self.filter_size] += (
                            grad_output[i, j, k, l] * self.filters[j]
                        )

        return grad_input


# Example usage
input_data = np.random.randn(1, 3, 28, 28)
conv_layer = Conv2D(num_filters=32, filter_size=3)
output = conv_layer.forward(input_data)
grad_output = np.random.randn(*output.shape)
grad_input = conv_layer.backward(grad_output)
