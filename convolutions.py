"""
matsjfunke
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def fold_arrays():
    """
    convoluting arrays a & b -> resulting in them smoothing
    """
    # Define the arrays
    a = np.array([0.1, 0.1, 1, 1, 1, 0.1, 0.1])
    b = np.array([0.2, 0.2, 0.2, 0.2])

    # Compute the convolution
    convolution = np.convolve(a, b, mode='full')
    print("Convolution:", convolution)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot array a
    axs[0].bar(np.arange(len(a)), a, color='red')
    axs[0].set_title('Array a')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Value')

    # Plot array b
    axs[1].bar(np.arange(len(b)), b, color='blue')
    axs[1].set_title('Array b')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Value')

    # Plot the convolution result
    axs[2].bar(np.arange(len(convolution)), convolution, color='green')
    axs[2].set_title('Convolution')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Value')

    # Show the plot
    plt.suptitle("Arrays A & B get folded / convoluted -> resulting in a smoothed version of both", fontsize=16)
    plt.tight_layout()
    plt.show()


def smooth_functions():
    """
    convoluting gaussian with f(x) to smooth it out
    """
    # Define multiple segments with different slopes
    def f(x):
        return np.where(x < -3, 0.5 * x + 4,
                        np.where(x < -1, -x - 1,
                                 np.where(x < 1, 0.5 * x + 1,
                                          np.where(x < 3, -x + 3, 0.5 * x - 2))))

    # Define a smoothing function (e.g., Gaussian)
    def gaussian_smooth(x, sigma=1.0):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    # Define the domain
    x = np.linspace(-5, 5, 500)

    # Compute the convolution (smoothing)
    smoothed = convolve(f(x), gaussian_smooth(x, sigma=1.0), mode='same')

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot original function f(x)
    axs[0].plot(x, f(x), color='red')
    axs[0].set_title('Original Function f(x)')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f(x)')

    # Plot smoothing function (Gaussian in this case)
    axs[1].plot(x, gaussian_smooth(x, sigma=1.0), color='green')
    axs[1].set_title('Smoothing Function (Gaussian)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Smoothing Kernel')

    # Plot smoothed function
    axs[2].plot(x, smoothed, color='blue')
    axs[2].set_title('Smoothed Function')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Smoothed f(x)')

    # Show the plot
    plt.suptitle("Convolution of Piecewise Linear Function with Gaussian Kernel", fontsize=16)
    plt.tight_layout()
    plt.savefig('images/smooth_function_convolution.png')
    plt.show()


def sharpen_functions():
    """
    edge detection highlights regions of rapid intensity change
    """
    # Define multiple segments with different slopes
    def f(x):
        return np.where(x < -3, 0.5 * x + 4,
                        np.where(x < -1, -x - 1,
                                 np.where(x < 1, 0.5 * x + 1,
                                          np.where(x < 3, -x + 3, 0.5 * x - 2))))

    # Define a Laplacian of Gaussian (LoG) function to magnify extremes
    def laplacian_of_gaussian(x, sigma=1.0):
        return (-1 / (np.pi * sigma**4)) * (1 - (x**2 / (2 * sigma**2))) * np.exp(-x**2 / (2 * sigma**2))

    # Define the domain
    x = np.linspace(-5, 5, 500)

    # Compute the convolution (magnifying the extremes)
    magnified = convolve(f(x), laplacian_of_gaussian(x, sigma=1.0), mode='same')

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Plot original function f(x)
    axs[0].plot(x, f(x), color='red')
    axs[0].set_title('Original Function f(x)')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('f(x)')

    # Plot Laplacian of Gaussian function (Kernel)
    axs[1].plot(x, laplacian_of_gaussian(x, sigma=1.0), color='green')
    axs[1].set_title('Magnifying Function (Laplacian of Gaussian)')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Magnifying Kernel')

    # Plot magnified function
    axs[2].plot(x, magnified, color='blue')
    axs[2].set_title('Magnified Function')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('Magnified f(x)')

    # Show the plot
    plt.suptitle("Convolution of Piecewise Linear Function with Laplacian of Gaussian Kernel", fontsize=16)
    plt.tight_layout()
    plt.savefig('images/sharp_function_convolution.png')
    plt.show()


if __name__ == "__main__":
    fold_arrays()
    smooth_functions()
    sharpen_functions()
