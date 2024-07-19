"""
matsjfunke
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def load_img(img_path):
    # Load the image and convert to grayscale
    img = Image.open(img_path)

    # Get image dimensions
    width, height = img.size
    print(f"Image dimensions: {width} x {height}")

    # Convert image to numpy array
    img_array = np.array(img)

    return img_array


def convolve_img(img_array, kernel):

    convolved_img = convolve(img_array, kernel)  # Apply convolution

    # Display original and convolved images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(convolved_img)
    axes[1].set_title("Image Edges")
    axes[1].axis("off")

    plt.show()


if __name__ == "__main__":
    img_path = "images/pixel-man.png"
    img_array = load_img(img_path)

    # kernel for edge detection
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    convolve_img(img_array, edge_kernel)
