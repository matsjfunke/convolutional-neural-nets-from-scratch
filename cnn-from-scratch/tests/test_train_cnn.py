"""
matsjfunke
"""

import unittest

import numpy as np
from train_cnn import ConvolutionalNeuralNetwork


class TestConvolutionalNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # Sample data and parameters
        self.input_shape = (8, 8)
        self.num_kernels = 2
        self.kernel_size = 3
        self.pooling_kernel_size = 2
        self.stride = 1
        self.hidden_layer_sizes = [10, 5]
        self.num_classes = 3

        # Initialize the network
        self.nn = ConvolutionalNeuralNetwork(
            input_shape=self.input_shape,
            num_kernels=self.num_kernels,
            kernel_size=self.kernel_size,
            pooling_kernel_size=self.pooling_kernel_size,
            stride=self.stride,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_classes=self.num_classes,
        )

        # Sample input and label
        self.input_img = np.random.randn(*self.input_shape)
        self.true_label = np.zeros(self.num_classes)
        self.true_label[1] = 1  # Assume class 1 is the true class
        self.learning_rate = 0.01

    def test_initialization(self):
        self.assertEqual(len(self.nn.kernels), self.num_kernels)
        self.assertEqual(len(self.nn.hidden_layers), len(self.hidden_layer_sizes))
        self.assertEqual(self.nn.output_weights.shape[0], self.hidden_layer_sizes[-1])
        self.assertEqual(self.nn.output_weights.shape[1], self.num_classes)

    def test_convolutional_layer(self):
        feature_maps = self.nn.convolutional_layer(self.input_img)
        self.assertEqual(len(feature_maps), self.num_kernels)
        self.assertEqual(feature_maps[0].shape[0], self.input_shape[0] - self.kernel_size + 1)
        self.assertEqual(feature_maps[0].shape[1], self.input_shape[1] - self.kernel_size + 1)

    def test_relu_layer(self):
        self.nn.convolutional_layer(self.input_img)
        relu_feature_maps = self.nn.conv_relu_layer()
        self.assertEqual(len(relu_feature_maps), self.num_kernels)
        self.assertTrue(np.all(relu_feature_maps[0] >= 0))

    def test_pooling_layer(self):
        self.nn.convolutional_layer(self.input_img)
        self.nn.conv_relu_layer()
        pool_feature_maps = self.nn.pooling_layer()
        self.assertEqual(len(pool_feature_maps), self.num_kernels)
        self.assertEqual(pool_feature_maps[0].shape[0], (self.input_shape[0] - self.kernel_size) // self.stride - self.pooling_kernel_size + 2)
        self.assertEqual(pool_feature_maps[0].shape[1], (self.input_shape[1] - self.kernel_size) // self.stride - self.pooling_kernel_size + 2)

    def test_forward_pass(self):
        probabilities, _, _, _, _ = self.nn.forward_pass(self.input_img)
        self.assertEqual(probabilities.shape[0], self.num_classes)
        self.assertAlmostEqual(np.sum(probabilities), 1.0, delta=1e-6)  # Probabilities should sum to 1

    def test_backward_pass(self):
        probabilities, _, _, _, _ = self.nn.forward_pass(self.input_img)
        loss = self.nn.backward_pass(self.input_img, probabilities, self.true_label, learning_rate=self.learning_rate)
        self.assertIsInstance(loss, float)

    def test_train(self):
        # Mock training data
        train_images = np.random.randn(10, *self.input_shape)
        train_labels = np.random.randint(0, self.num_classes, size=10)

        # Override method to prevent printing during tests
        def dummy_train(self, train_images, train_labels, num_epochs=1, batch_size=2, learning_rate=0.001):
            num_samples = train_images.shape[0]
            for _ in range(num_epochs):
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
                        probabilities, _, _, _, _ = self.forward_pass(input_img)

                        # Reset accumulated gradients
                        self.kernels_grads = [np.zeros_like(kernel) for kernel in self.kernels]

                        # Compute loss and update gradients
                        loss = self.backward_pass(input_img, probabilities, true_label, learning_rate)

                        # Track loss and accuracy
                        epoch_loss += loss
                        if np.argmax(probabilities) == true_label:
                            correct_predictions += 1

                average_loss = epoch_loss / num_batches
                accuracy = correct_predictions / num_samples
                return average_loss, accuracy

        self.nn.train = dummy_train.__get__(self.nn)
        avg_loss, accuracy = self.nn.train(train_images, train_labels, num_epochs=1, batch_size=2, learning_rate=self.learning_rate)
        self.assertTrue(0 <= accuracy <= 1)

    def test_save_load_params(self):
        self.nn.save_params("test_params.pkl")
        nn_loaded = ConvolutionalNeuralNetwork(
            input_shape=self.input_shape,
            num_kernels=self.num_kernels,
            kernel_size=self.kernel_size,
            pooling_kernel_size=self.pooling_kernel_size,
            stride=self.stride,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_classes=self.num_classes,
        )
        nn_loaded.load_params("test_params.pkl")

        # Check if loaded parameters are the same as saved parameters
        np.testing.assert_array_equal(self.nn.kernels, nn_loaded.kernels)
        for (weights, biases), (weights_loaded, biases_loaded) in zip(self.nn.hidden_layers, nn_loaded.hidden_layers):
            np.testing.assert_array_equal(weights, weights_loaded)
            np.testing.assert_array_equal(biases, biases_loaded)
        np.testing.assert_array_equal(self.nn.output_weights, nn_loaded.output_weights)
        np.testing.assert_array_equal(self.nn.output_biases, nn_loaded.output_biases)


if __name__ == "__main__":
    unittest.main()
