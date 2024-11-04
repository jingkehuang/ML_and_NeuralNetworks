'''
Assignment 2 starter code for CPSC 383
Fall 2024

Works on the heart.csv data set
Goal is to compare different variations on a basic neural net in TensorBoard and write an analysis of the results

Author: Jingke Huang
UCID: 30115284
Date: Nov 3, 2024
Course: CPSC 383
Semester: Fall 2024
Tutorial: T02

Additional tutorials or resources I used to help build this model:
https://cspages.ucalgary.ca/~colton.gowans/cpsc383/
'''

# Removes extra log messages (change if you like)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1_l2
import datetime

# TensorBoard logs are saved in the following location:
tf.compat.v1.enable_eager_execution()
# log_dir_base = "logs/fit/"  # Base directory for TensorBoard logs
log_dir_base = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

###################################################
'''
The prepData() function reads and encodes data from the heart.csv file.
It splits the entries into training and testing sets and returns two tuples,
(x_train, y_train) and (x_test, y_test).
'''
def readData(filename):
    data = pd.read_csv(filename)
    features = data.iloc[:, :-1]  # All columns except the last one
    labels = data.iloc[:, -1]     # The last column (chd)

    # Convert categorical features
    features['famhist'] = features['famhist'].apply(lambda x: 1 if x == 'Present' else 0)

    # Normalize age to have a max value of 1
    features['age'] = features['age'] / features['age'].max()

    features = features.astype(float) 

    # Standardize other columns
    scaler = StandardScaler()
    features.iloc[:, :-1] = scaler.fit_transform(features.iloc[:, :-1])  # Exclude the label column from scaling

    return len(data), features, labels

def prepData():
    n, features, labels = readData("heart.csv")

    # Split data into training and testing sets (5:1 ratio)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1/6, random_state=42)

    # Convert to numpy arrays
    trainingFeatures, testingFeatures = X_train.to_numpy(), X_test.to_numpy()
    trainingLabels, testingLabels = y_train.to_numpy(), y_test.to_numpy()

    print(f"Number of training samples: {trainingFeatures.shape[0]}")
    print(f"Number of testing samples: {testingFeatures.shape[0]}")

    return (trainingFeatures, trainingLabels), (testingFeatures, testingLabels)

###################################################

(x_train, y_train), (x_test, y_test) = prepData()


def build_and_train_model(model, log_dir, epochs=30, learning_rate=0.001):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard_callback], validation_split=0.1)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")


# Baseline Model
def build_baseline_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Added dropout to reduce overfitting
        tf.keras.layers.Dense(1, activation='sigmoid')  # Single output for binary classification
    ])
    return model

# Variant 1: Increase neurons and add dropout
def build_variant_1():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Variant 2: Add Batch Normalization and dropout with increased layer size
def build_variant_2():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Variant 3: Increase the number of layers and add kernel regularization
def build_variant_3():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Variant 4: Deeper model with smaller layers, dropout and regularization
def build_variant_4():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dropout(0.1),  # Adjust dropout rates
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Run each model with its own log directory
baseline_model = build_baseline_model()
build_and_train_model(baseline_model, log_dir_base + "baseline", epochs=30)

variant_1_model = build_variant_1()
build_and_train_model(variant_1_model, log_dir_base + "variant_1", epochs=30)

variant_2_model = build_variant_2()
build_and_train_model(variant_2_model, log_dir_base + "variant_2", epochs=30)

variant_3_model = build_variant_3()
build_and_train_model(variant_3_model, log_dir_base + "variant_3", epochs=40)  # Increased epochs

variant_4_model = build_variant_4()
build_and_train_model(variant_4_model, log_dir_base + "variant_4", epochs=50)  # Increased epochs
