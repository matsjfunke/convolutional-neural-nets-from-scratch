"""
matsjfunke

go to https://www.cs.toronto.edu/~kriz/cifar.html and download https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
expand it and move cifar-10-batches-py into images folder
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def reshape_data(data):
    reshaped_data = data.reshape(-1, 3, 32, 32)
    reshaped_data = reshaped_data.transpose(0, 2, 3, 1)
    return reshaped_data


def load_all_batches(path):
    data_batches = []
    labels_batches = []
    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f"data_batch_{i}"))
        data_batches.append(batch[b"data"])
        labels_batches.append(batch[b"labels"])

    data = np.concatenate(data_batches)
    labels = np.concatenate(labels_batches)
    return reshape_data(data), labels


def load_cifar10(path):
    train_images, train_labels = load_all_batches(path)
    test_batch = unpickle(os.path.join(path, "test_batch"))
    test_images = reshape_data(test_batch[b"data"])
    test_labels = test_batch[b"labels"]
    meta = unpickle(os.path.join(path, "batches.meta"))
    label_names = [label.decode("utf-8") for label in meta[b"label_names"]]
    return train_images, train_labels, test_images, test_labels, label_names


def plot_cifar_images():
    indices = [19, 1, 28, 4, 13, 7, 8, 9, 49, 56]
    # Create a figure
    plt.figure(figsize=(15, 6))  # Adjust the size as needed

    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)  # 2 rows, 5 columns
        plt.imshow(train_images[index])
        plt.axis("off")
        plt.title(f"Label: {label_names[train_labels[index]]}")

    plt.tight_layout()
    plt.savefig("images/cifar-10-labels.png")
    plt.show()


if __name__ == "__main__":
    # Set the path to the directory containing the CIFAR-10 batches
    path = "images/cifar-10-batches-py"

    print("CIFAR-10 dataset")
    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(
        path
    )
    print(
        f"Set contains: {train_images.shape[0] + test_images.shape[0]} images of {label_names} categories"
    )

    # plot image from train_images given index
    plot_cifar_images()
