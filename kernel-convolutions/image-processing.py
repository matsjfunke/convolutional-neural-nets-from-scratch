"""
matsjfunke
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def load_img(img_path):
    # Load the image as a NumPy array
    img = mpimg.imread(img_path)

    height, width, channels = img.shape
    print(f"Image dimensions: {width} x {height}")
    print(f"Number of channels: {channels}")

    x, y = 81, 35
    pixel_value = img[y, x]  # Access pixel value at (x, y)
    if channels == 4:  # RGBA image
        red, green, blue, alpha = pixel_value
        print(f"\nThe pixel @ x={x} & y={y} has values:")
        print(f"Red: {red}")
        print(f"Green: {green}")
        print(f"Blue: {blue}")
        print(f"Alpha/Opacity: {alpha}")
    elif channels == 3:  # RGB image
        red, green, blue = pixel_value
        print(f"\nThe pixel @ x={x} & y={y} has values:")
        print(f"Red: {red}")
        print(f"Green: {green}")
        print(f"Blue: {blue}")
    return img


def convolve_img(img_array, kernel):
    print("\nconvoluting...")
    kernel = np.array(kernel)
    kernel_height, kernel_width = kernel.shape
    height, width, channels = img_array.shape

    # initialize output image to zeros
    output_img = np.zeros_like(img_array)

    # Padding to handle edges
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image to handle edges
    padded_img = np.pad(img_array, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode="constant")

    for y in range(height):
        for x in range(width):
            for c in range(channels):
                region = padded_img[y : y + kernel_height, x : x + kernel_width, c]
                output_img[y, x, c] = np.sum(region * kernel)

    print("convolution done")
    return output_img


def plot_img_convolution(original_img, convolved_img):
    print("\nploting...")
    # Display original and convolved images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("on")
    axes[0].grid(True)

    axes[1].imshow(convolved_img)
    axes[1].set_title("Convolved Image")
    axes[1].axis("on")
    axes[1].grid(True)

    plt.show()


if __name__ == "__main__":
    img_path = "images/cafe-dog.png"
    # img_path = "images/skydive-plane.png"
    img_array = load_img(img_path)

    # kernel for edge detection
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # convolved_img = convolve_img(img_array, edge_kernel)
    # plot_img_convolution(img_array, convolved_img)

    # kernel for blurring
    gaussian_kernel = np.array(
        [
            [1, 4, 7, 4, 1],
            [4, 16, 26, 16, 4],
            [7, 26, 41, 26, 7],
            [4, 16, 26, 16, 4],
            [1, 4, 7, 4, 1],
        ]
    )
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
    mean_kernel = np.array(
        [
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    # convolved_img = convolve_img(img_array, mean_kernel)
    # plot_img_convolution(img_array, convolved_img)

    # kernels for brightening /darkening
    brightening_kernel = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
    darkening_kernel = np.array([[0, 0, 0], [0, 0.2, 0], [0, 0, 0]])
    convolved_img = convolve_img(img_array, darkening_kernel)
    plot_img_convolution(img_array, convolved_img)
