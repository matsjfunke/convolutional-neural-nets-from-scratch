"""
matsjfunke
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model("cifar10_cnn_model.h5")

# Load and preprocess the input image
# input_img = ["images/cafe-dog.png", "dog"]  # Replace with the path to your image & label
input_img = ["images/skydive-plane.png", "airplane"]
img = image.load_img(input_img[0], target_size=(32, 32))

# Convert image to numpy array and preprocess
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict the class probabilities
predictions = model.predict(img_array)

# Get the predicted class index and label
predicted_class = np.argmax(predictions, axis=1)[0]
label_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
predicted_label = label_names[predicted_class]

# Plot the image pixels
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis("off")
plt.title(f"Image labeled as {input_img[1]}, predicted: {predicted_label}")
plt.show()
