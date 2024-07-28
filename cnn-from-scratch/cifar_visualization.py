"""
matsjfunke

"""

import matplotlib.pyplot as plt
from cifar_10_utils import load_cifar10


def plot_cifar_images():
    indices = [19, 1, 28, 4, 13, 7, 8, 9, 49, 56]
    plt.figure(figsize=(15, 6))

    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
        plt.imshow(train_images[index])
        plt.axis("off")
        plt.title(f"Label: {label_names[train_labels[index]]}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = "../images/cifar-10-batches-py"

    print("CIFAR-10 dataset")
    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)
    print(f"Set contains: {train_images.shape[0] + test_images.shape[0]}\nimages of categories: {label_names}\nimages are {test_images.shape[1]}:{test_images.shape[2]} pixels")

    plot_cifar_images()
