'''
Assignment 2 starter code for CPSC 383
Fall 2024

Works on the heart.csv data set
Goal is to compare different variations on a basic neural net
in TensorBoard and write an analysis of the results

Author: Jingke Huang
30115284
Nov 3, 2024

'''

# removes extra log messages (change if you like)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorBoard logs are saved in the following location:
log_dir = "logs/fit/run_name"  # replace run_name with a different identifier for each net your try
# log_dir_base = "logs/fit/"  # Base directory for TensorBoard logs

###################################################
'''
The prepData() function should read in and encode data from the heart.csv file.
You will need to encode the features yourself according to the instructions in the assignment.
You will also need to split the entries in the file into training and testing sets.

The function should return two tuples, (x_train, y_train) and (x_test, y_test)

- x_train and x_test should be 2D numpy arrays whose rows are the encoded feature vectors
for the training and testing data, respectively.

- y_train and y_test should be 1D numpy arrays whose entries are the 0-1 labels for the corresponding data points

'''


def readData(filename):
    data = pd.read_csv(filename)
    features = data.iloc[:, :-1]  # All columns except the last one
    labels = data.iloc[:, -1]     # The last column (chd)

    # Convert categorical features
    features['famhist'] = features['famhist'].apply(lambda x: 1 if x == 'Present' else 0)

    # Normalize age to have a max value of 1
    features['age'] = features['age'] / features['age'].max()

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

print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)



# # build our neural net
# model = tf.keras.models.Sequential([
#     tf.keras.Input(shape=(x_train.shape[1],)), # Dynamic shape based on features
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')  # Single output for binary classification
# ])
# print(model.summary())


# loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # Use binary crossentropy for binary output


# model.compile(optimizer = tf.keras.optimizers.Adam(),
#                 loss=loss_fn,
#                 metrics = ['accuracy']) 

# # add records to tensorboard
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# # train and then test the model
# model.fit(x_train, y_train, epochs = 20, callbacks = [tensorboard_callback])
# model.evaluate(x_test, y_test, verbose = 2)

# Function to build and train models
def build_and_train_model(model, log_dir, epochs=20):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train the model
    model.fit(x_train, y_train, epochs=epochs, callbacks=[tensorboard_callback])
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

# Baseline Model
def build_baseline_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Single output for binary classification
    ])
    return model

# Variant 1: Increase neurons and add dropout
def build_variant_1():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Variant 2: Add Batch Normalization
def build_variant_2():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),  # Add batch normalization
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Variant 3: Increase layers and epochs
def build_variant_3():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),  # Add another layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Variant 4: Add kernel regularization
def build_variant_4():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # Add regularization
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Run each model with its own log directory
baseline_model = build_baseline_model()
build_and_train_model(baseline_model, log_dir_base + "baseline", epochs=20)

variant_1_model = build_variant_1()
build_and_train_model(variant_1_model, log_dir_base + "variant_1", epochs=20)

variant_2_model = build_variant_2()
build_and_train_model(variant_2_model, log_dir_base + "variant_2", epochs=20)

variant_3_model = build_variant_3()
build_and_train_model(variant_3_model, log_dir_base + "variant_3", epochs=30)  # Increased epochs

variant_4_model = build_variant_4()
build_and_train_model(variant_4_model, log_dir_base + "variant_4", epochs=20)
