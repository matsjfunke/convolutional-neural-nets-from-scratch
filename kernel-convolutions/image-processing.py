"""
matsjfunke

Positive kernel values make areas appear lighter (white) in the output image.
Negative kernel values make areas appear darker (black) in the output image.
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
    print("convoluting...")
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
    print("ploting...")
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




def horizontal_edge_detection(img_array, plot=True):
    print("\ndetecting horizontal edges")
    sobel_horizontal_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    convolved_img = convolve_img(img_array, sobel_horizontal_kernel)
    if plot:
        plot_img_convolution(img_array, convolved_img)
    return convolved_img


def vertical_edge_detection(img_array, plot=True):
    print("\ndetecting vertical edges")
    sobel_vertical_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    convolved_img = convolve_img(img_array, sobel_vertical_kernel)
    if plot:
        plot_img_convolution(img_array, convolved_img)
    return convolved_img


def diagonal_change_detection(img_array, kernel_type="bottom_left_to_top_right", plot=True):
    print("\nDetecting diagonals")

    kernels = {
        "bottom_left_to_top_right": np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]]),
        "top_left_to_bottom_right": np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
        "bottom_right_to_top_left": np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
        "top_right_to_bottom_left": np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]),
    }
    if kernel_type not in kernels:
        raise ValueError(f"Invalid kernel_type '{kernel_type}'. Valid options are: {list(kernels.keys())}")
    diagonal_kernel = kernels[kernel_type]

    convolved_img = convolve_img(img_array, diagonal_kernel)

    if plot:
        plot_img_convolution(img_array, convolved_img)

    return convolved_img


def edge_detection(img_array, plot=True):
    print("\ndetecting edges")
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    convolved_img = convolve_img(img_array, edge_kernel)
    if plot:
        plot_img_convolution(img_array, convolved_img)
    return convolved_img
def img_blurring(img_array, plot=True, blur_type="gaussian"):
    print("\nblurring the image")
    if blur_type == "gaussian":
        gaussian_kernel = np.array(
            [
                [1, 4, 7, 4, 1],
                [4, 16, 26, 16, 4],
                [7, 26, 41, 26, 7],
                [4, 16, 26, 16, 4],
                [1, 4, 7, 4, 1],
            ]
        )
        kernel = gaussian_kernel / np.sum(gaussian_kernel)
    elif blur_type == "mean":
        kernel = np.array(
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        )
    else:
        print("blurrtype option doesnt exist")

    convolved_img = convolve_img(img_array, kernel)
    if plot:
        plot_img_convolution(img_array, convolved_img)

    return convolved_img


def brighten_darken(img_array, plot=True, mode="dark"):
    # kernels for brightening /darkening
    if mode == "dark":
        print("darkening")
        darkening_kernel = np.array([[0, 0, 0], [0, 0.2, 0], [0, 0, 0]])
        convolved_img = convolve_img(img_array, darkening_kernel)
        if plot:
            plot_img_convolution(img_array, convolved_img)
        return convolved_img
    elif mode == "bright":
        print("brightening")
        brightening_kernel = np.array([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
        convolved_img = convolve_img(img_array, brightening_kernel)
        if plot:
            plot_img_convolution(img_array, convolved_img)
        return convolved_img
    else:
        print("invalid parameter for option, use 'dark' or 'bright'")


def pool_img(img_array, plot=True, pool_size=(2, 2), pool_type="max"):
    print("\npooling image")
    height, width, channels = img_array.shape
    pooled_height = height // pool_size[0]
    pooled_width = width // pool_size[1]
    pooled_img = np.zeros((pooled_height, pooled_width, channels))

    for y in range(pooled_height):
        for x in range(pooled_width):
            for c in range(channels):
                region = img_array[y * pool_size[0] : (y + 1) * pool_size[0], x * pool_size[1] : (x + 1) * pool_size[1], c]
                if pool_type == "max":
                    pooled_img[y, x, c] = np.max(region)
                elif pool_type == "average":
                    pooled_img[y, x, c] = np.mean(region)
                else:
                    raise ValueError("Invalid pool_type. Use 'max' or 'average'.")

    if plot:
        plot_img_convolution(img_array, pooled_img)
    return pooled_img


if __name__ == "__main__":
    # img_path = "images/cafe-dog.png"
    img_path = "images/skydive-plane.png"
    img_array = load_img(img_path)

    vertical_edge_detection(img_array)

    horizontal_edge_detection(img_array)

    diagonal_change_detection(img_array, kernel_type="top_right_to_bottom_left", plot=True)

    result = edge_detection(img_array)

    img_blurring(img_array, plot=True, blur_type="mean")

    brighten_darken(img_array, plot=True, mode="dark")

    pool_img(img_array, plot=True, pool_size=(20, 20), pool_type="max")
    pooled_img = pool_img(result, plot=True, pool_size=(20, 20), pool_type="max")
