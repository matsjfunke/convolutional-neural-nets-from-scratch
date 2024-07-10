"""
matsjfunke
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models


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


# Function to build CNN model
def build_cnn_model():
    """
    Function to build a CNN model for CIFAR-10 classification.

    Returns:
    - Compiled Keras Sequential model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    return model


# Function to compile, train, and evaluate the CNN model
def cnn_for_cifar(train_images, train_labels, test_images, test_labels, epochs=10):
    """
    Function to train a CNN model on CIFAR-10 dataset.

    Parameters:
    - train_images: 4D array of training images with shape (N, 32, 32, 3).
    - train_labels: 1D array of training labels.
    - test_images: 4D array of test images with shape (N_test, 32, 32, 3).
    - test_labels: 1D array of test labels.
    - epochs: Number of epochs to train the model.

    Returns:
    - Tuple of (model, history) where:
      - model: Trained Keras Sequential model.
      - history: Training history.
    """
    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Ensure the labels are in the correct shape
    train_labels = np.array(train_labels).reshape(-1)
    test_labels = np.array(test_labels).reshape(-1)

    model = build_cnn_model()

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")
    return model, history


if __name__ == "__main__":
    # Set the path to the directory containing the CIFAR-10 batches
    path = "../images/cifar-10-batches-py"

    # Load CIFAR-10 data
    train_images, train_labels, test_images, test_labels, label_names = load_cifar10(
        path
    )
    print(
        f"Set contains: {train_images.shape[0] + test_images.shape[0]} images of {label_names} categories"
    )

    # Train and evaluate CNN model for CIFAR-10
    model, history = cnn_for_cifar(
        train_images, train_labels, test_images, test_labels, 10
    )

    # save the models parameters
    model.save("./cifar10_cnn_model.h5")

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # Plot training & validation accuracy values
    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["val_accuracy"])
    ax1.set_title("Model accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"], loc="upper left")

    # Plot training & validation loss values
    ax2.plot(history.history["loss"])
    ax2.plot(history.history["val_loss"])
    ax2.set_title("Model loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"], loc="upper left")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
