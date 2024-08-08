import unittest

import numpy as np
from neural_net_utils import (
    calc_conv_output_size,
    convolve2d,
    cross_entropy_loss_gradient,
    init_weights_biases,
    max_pooling,
    relu,
    relu_derivative,
    softmax,
)


class TestNeuralNetUtils(unittest.TestCase):

    def setUp(self):
        """
        Set up the test data.
        """
        self.input_img_array = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 2], [0, 1, 2, 3]])

        self.kernels = [np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])]

        # Expected outputs based on provided values
        self.expected_outputs = {0: np.array([[4, 4, -2], [4, 4, -4], [-6, -6, -6]]), 1: np.array([[2, 2, 6], [2, 2, 8], [-8, -8, 0]])}

    def test_init_weights_biases(self):
        num_inputs, num_outputs = 4, 3
        weights, biases = init_weights_biases(num_inputs, num_outputs)
        self.assertEqual(weights.shape, (num_inputs, num_outputs))
        self.assertEqual(biases.shape, (num_outputs,))
        self.assertTrue(np.all(biases == 0))

    def test_convolution(self):
        """
        Test that the manual convolution matches the expected outputs.
        """
        for idx, kernel in enumerate(self.kernels):
            feature_maps_manual = convolve2d(self.input_img_array, kernel)
            expected_output = self.expected_outputs[idx]
            self.assertTrue(
                np.allclose(feature_maps_manual, expected_output),
                f"Manual convolution result for kernel {idx} does not match the expected output.\n"
                f"Computed:\n{feature_maps_manual}\nExpected:\n{expected_output}",
            )

    def test_max_pooling(self):
        feature_map = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 2], [0, 1, 2, 3]])
        expected_output = np.array([[5, 6], [8, 9]])
        pooled_feature_map = max_pooling(feature_map)
        np.testing.assert_array_equal(pooled_feature_map, expected_output)

    def test_calc_conv_output_size(self):
        input_size = (32, 32)
        kernel_size = 3
        pooling_kernel_size = 2
        stride = 2
        num_kernels = 64
        expected_output = 64 * ((32 - 3 + 1 - 2) // 2 + 1) ** 2
        output_size = calc_conv_output_size(input_size, kernel_size, pooling_kernel_size, stride, num_kernels)
        self.assertEqual(output_size, expected_output)

    def test_relu(self):
        x = np.array([-1, 0, 1, 2])
        expected_output = np.array([0, 0, 1, 2])
        result = relu(x)
        np.testing.assert_array_equal(result, expected_output)

    def test_relu_derivative(self):
        x = np.array([-1, 0, 1, 2])
        expected_output = np.array([0, 0, 1, 1])
        result = relu_derivative(x)
        np.testing.assert_array_equal(result, expected_output)

    def test_softmax(self):
        logits = np.array([1.0, 2.0, 3.0])
        expected_output = np.array([0.09003057, 0.24472847, 0.66524096])
        result = softmax(logits)
        np.testing.assert_almost_equal(result, expected_output, decimal=6)

    def test_softmax_with_nan_inf(self):
        logits = np.array([np.nan, np.inf])
        with self.assertRaises(ValueError):
            softmax(logits)

    def test_cross_entropy_loss_gradient(self):
        true_labels = np.array([0, 1, 0])
        predicted_probs = np.array([0.1, 0.8, 0.1])
        loss, gradient = cross_entropy_loss_gradient(true_labels, predicted_probs)
        expected_loss = -np.sum(true_labels * np.log(predicted_probs))
        expected_gradient = predicted_probs - true_labels
        self.assertAlmostEqual(loss, expected_loss)
        np.testing.assert_array_almost_equal(gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
