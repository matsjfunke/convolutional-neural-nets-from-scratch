"""
matsjfunke

28.07.2024

go to https://www.cs.toronto.edu/~kriz/cifar.html and download https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
expand it and move cifar-10-batches-py into images folder
"""

import os
import pickle

import numpy as np


# Functions to load and preprocess CIFAR-10 data
def unpickle(file):
    """
    Function to unpickle a CIFAR-10 batch file.

    Parameters:
    - file: Path to the CIFAR-10 batch file.

    Returns:
    - Dictionary containing the data from the batch file.
    """
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def reshape_data(data):
    """
    Function to reshape CIFAR-10 data from flat to 3D arrays.

    Parameters:
    - data: Flat data array with shape (N, 3072).

    Returns:
    - Reshaped data array with shape (N, 32, 32, 3).
    """
    reshaped_data = data.reshape(-1, 3, 32, 32)
    reshaped_data = reshaped_data.transpose(0, 2, 3, 1)
    return reshaped_data


def load_all_batches(path):
    """
    Function to load all CIFAR-10 batches.

    Parameters:
    - path: Directory path containing CIFAR-10 batch files.

    Returns:
    - Tuple of (train_images, train_labels) where:
      - train_images: 4D array of training images with shape (N, 32, 32, 3).
      - train_labels: 1D array of training labels.
    """
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
    """
    Function to load CIFAR-10 dataset.

    Parameters:
    - path: Directory path containing CIFAR-10 batch files.

    Returns:
    - Tuple of (train_images, train_labels, test_images, test_labels, label_names) where:
      - train_images: 4D array of training images with shape (N, 32, 32, 3).
      - train_labels: 1D array of training labels.
      - test_images: 4D array of test images with shape (N_test, 32, 32, 3).
      - test_labels: 1D array of test labels.
      - label_names: List of label names corresponding to CIFAR-10 categories.
    """
    train_images, train_labels = load_all_batches(path)
    test_batch = unpickle(os.path.join(path, "test_batch"))
    test_images = reshape_data(test_batch[b"data"])
    test_labels = np.array(test_batch[b"labels"])
    meta = unpickle(os.path.join(path, "batches.meta"))
    label_names = [label.decode("utf-8") for label in meta[b"label_names"]]
    return train_images, train_labels, test_images, test_labels, label_names


if __name__ == "__main__":
    # Set the path to the directory containing the CIFAR-10 batches
    path = "../images/cifar-10-batches-py"

    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(path)
    print(f"Set contains: {train_images.shape[0] + test_images.shape[0]} images of {label_names} categories")
