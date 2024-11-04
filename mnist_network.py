"""
CPSC 383, Fall 2024

Neural network for processing the MNIST dataset. 
This code trains a neural network model to achieve 95%+ accuracy on the MNIST dataset and saves the model.
Based on TensorFlow tutorials: https://www.tensorflow.org/tutorials/quickstart/beginner

Author: Jingke Huang
Date: Nov 2, 2024
UCID: 30115284
Course: CPSC 383
Semester: Fall 2024
Tutorial: T02

Additional tutorials or resources I used to help build this model:
https://cspages.ucalgary.ca/~colton.gowans/cpsc383/

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

print("TensorFlow version:", tf.__version__)

# Converts 1D vector into 2D column vector
def cv(vec):
    return np.array([vec]).T

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
path = "mnist.npz"  # Adjust to your path if needed

# Try to load the dataset, fallback to local if the URL fetch fails
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
except Exception as e:
    print("Error loading data, using local file instead:", e)
    # Check if local file exists
    if os.path.exists(path):
        with np.load(path) as data:
            x_train, y_train = data['x_train'], data['y_train']
            x_test, y_test = data['x_test'], data['y_test']
    else:
        raise FileNotFoundError("Local file not found. Please download mnist.npz.")

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to 2D column vectors for easier handling
y_train = cv(y_train)
y_test = cv(y_test)

# Plot an example from the MNIST dataset
print("Label for the first training sample:", y_train[0][0])
plt.imshow(x_train[0], cmap="gray")  # Display in grayscale
plt.title(f"Training Sample - Label: {y_train[0][0]}")
plt.show()

# Create a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28)),       # Input layer expects 28x28 images
    tf.keras.layers.Flatten(),            # Flatten the 2D images to 1D vectors
    tf.keras.layers.Dense(128, activation="relu"),  # Dense hidden layer with ReLU activation
    tf.keras.layers.Dropout(0.2),         # Dropout for regularization
    tf.keras.layers.Dense(10)             # Output layer with 10 units (one for each digit)
])

# Compile the model with loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=loss_fn,
              metrics=["accuracy"])

# Set up TensorBoard for optional tracking
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")


model.save("mnist_network.h5")
print("Model saved as 'mnist_network.h5'.")

# Load handwritten digits from the saved .npy file
try:
    handwritten_digits = np.load("digits.npy")
    handwritten_digits = handwritten_digits / 255.0  # Normalize as well
except FileNotFoundError:
    print("Error: digits.npy file not found.")
    exit()

# Loop to display each handwritten digit and show predictions
for i in range(10):
    digit_image = handwritten_digits[i]
    digit_image = np.expand_dims(digit_image, axis=0) 
    
    # Get model predictions
    predictions = model(digit_image).numpy()
    probs = tf.nn.softmax(predictions).numpy()  
    
    # Display the handwritten digit and the predicted label
    plt.imshow(handwritten_digits[i], cmap="gray")
    plt.title(f"Handwritten Digit {i} - Predicted Label: {np.argmax(probs[0])}")
    plt.show()
    
    print(f"Output probabilities for digit {i}: {probs[0]}")
    print(f"Model's predicted label: {np.argmax(probs[0])}\n")
