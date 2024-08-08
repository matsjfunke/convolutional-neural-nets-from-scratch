import numpy as np
from scipy.signal import convolve2d


def convolve2d_manual(img, kernel):
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


class ConvolutionalLayer:
    def __init__(self, kernels):
        self.kernels = kernels
        self.conv_feature_maps = []

    def convolutional_layer(self, input_img_array):
        """
        Apply convolution to the input image with kernels to create feature_maps.
        """
        feature_maps = [convolve2d_manual(input_img_array, kernel) for kernel in self.kernels]
        self.conv_feature_maps = feature_maps
        return feature_maps


# Example usage
if __name__ == "__main__":
    # Define a sample image and kernels
    input_img_array = np.array([[1, 2, 3, 0], [4, 5, 6, 1], [7, 8, 9, 2], [0, 1, 2, 3]])

    kernels = [np.array([[1, 0], [0, -1]]), np.array([[0, 1], [-1, 0]])]

    # Create a convolutional layer instance with the defined kernels
    conv_layer = ConvolutionalLayer(kernels)

    # Apply convolutional layer to the input image
    feature_maps_manual = conv_layer.convolutional_layer(input_img_array)

    # Compare with scipy's convolution2d
    feature_maps_scipy = [convolve2d(input_img_array, kernel, mode="valid") for kernel in kernels]

    # Output the feature maps from the manual implementation
    print("Feature maps (manual implementation):")
    for idx, fmap in enumerate(feature_maps_manual):
        print(f"Feature map {idx}:\n{fmap}\n")

    # Output the feature maps from the scipy implementation
    print("Feature maps (scipy implementation):")
    for idx, fmap in enumerate(feature_maps_scipy):
        print(f"Feature map {idx}:\n{fmap}\n")

    # Check if the results match
    for idx, (fmap_manual, fmap_scipy) in enumerate(zip(feature_maps_manual, feature_maps_scipy)):
        if np.allclose(fmap_manual, fmap_scipy):
            print(f"Feature map {idx} matches the scipy implementation.")
        else:
            print(f"Feature map {idx} does not match the scipy implementation.")
