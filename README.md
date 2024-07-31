# Convolutional Neural Networks (CNNs)

- [What are Convolutional Neural Networks?](#what-are-convolutional-neural-networks)
- [Why use Convolutional Neural Networks?](#why-use-convolutional-neural-networks)
- [Structure of a Convolutional Layer](#structure-of-a-convolutional-layer)
- [Backpropagation in a Convolutional Layer](#backpropagation-in-a-convolutional-layer)

## What are Convolutional Neural Networks?

Convolutional Neural Networks (CNNs) differ from normal feed-forward neural networks [more about FCNNs](https://github.com/matsjfunke/feedforward-neural-network-from-scratch) in that they include convolutional layers.
Convolutional layers apply a set of learnable filters to input data to extract local features by performing convolutions [click here to learn about kernel-convolutions](kernel-convolutions)

## Why use Convolutional Neural Networks?

Advantages of using convolutions as input:

1. Applying convolutions reduces the number of parameters, which enhances computational efficiency and mitigates the risk of overfitting.
   - In a convolutional layer, the network learns the values of the filters (or kernels, often 3x3), which are much smaller than the input images (e.g., JPEG, PNG). These filters have shared weights, meaning the same weights are applied across different spatial locations in the input. Unlike in fully connected neural networks where each neuron has its own set of unique weights. This sharing of weights in CNNs results in significantly fewer parameters, allowing the model to efficiently learn while being computationally more efficient and less prone to overfitting.
2. Using kernels allows detection of edges, textures, and shapes while preserving their spatial arrangement within the image.
   - In a [Fully Connected Neural Networks (FCNNs)](https://github.com/matsjfunke/feedforward-neural-network-from-scratch) the input image is flattened into a one-dimensional vector, losing its spatial structure. In contrast, Convolutional Neural Networks (CNNs use kernels to scan and process localized regions of the image, preserving the spatial relationships between features and retaining the original structure of the input data.

## Structure of a Convolutional layer

- Convolutional Layer: Core building block of a CNN. It consists of multiple filters (or kernels) that slide over the input image to produce feature maps.

- Kernel (or Filter): Small, learnable matrix used to perform convolutions on the input data. Each kernel extracts different features (e.g., edges, textures) by sliding over the input and applying convolutions, which involves element-wise multiplications and summations between the kernel and corresponding input regions. [more detail / visualisations here](kernel-convolutions)

- Feature Map: Output from applying a kernel to the input image, with a bias term added to adjust the result. Each feature map represents a specific feature detected by its corresponding kernel and captures the spatial arrangement of that feature in the input.

- ReLU (Rectified Linear Unit): Activation function applied to feature maps to introduce non-linearity into the model.

- Pooling: Down-sampling technique used to reduce the spatial dimensions of the feature maps (e.g., using max pooling or average pooling). Decreases the computational load and makes the network more robust to small translations and distortions in the input data. Think making it pixely / reducing image sharpness).

- Flattening: Converting the multi-dimensional feature maps into a one-dimensional vector. Necessary before fully connected layers, which require a flat input.

## Backpropagation in a Convolutional Layer

- How the CNN learns / updates its parameters: During backpropagation, the gradients of the loss function are propagated backward through the network, starting from the output layer.

- Gradient Feature Map: Calculating how the error (or loss) changes with respect to each feature map. The gradients indicate how significantly each feature map or kernel contributes to the loss, guiding the updates to reduce the loss most effectively, telling us how the feature maps should be adjusted to minimize the loss.

- Gradient of the Loss Function with Respect to Kernels (Filters): Computed using the gradients of the feature maps. By convolving the gradient of the loss with the input image (or the previous layer’s output), it measures how much each kernel contributed to the error and thus needs to be adjusted to improve the model's performance.

- Gradient of the Loss Function with Respect to Biases: Each kernel in a convolutional layer typically has an associated bias term. The gradient of the loss with respect to these biases is calculated by summing up the gradients of the feature maps over the spatial dimensions. This tells us how to adjust the biases to reduce the error.

- Weight Update: Using the computed gradients, the weights (kernels) and biases are updated using an optimization algorithm like stochastic gradient descent (SGD).

## Usage of the scripts

```bash
# Clone the repository
git clone https://github.com/matsjfunke/convolutional-neural-nets-from-scratch

# Install dependencies
pip install -r requirements.txt

# cd into directory
cd kernel-convolutions

# run scripts
python convolutions.py
```
