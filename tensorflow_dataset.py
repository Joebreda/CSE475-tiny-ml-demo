import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


import tensorflow as tf

import utils 

PLOT = False

def get_model(epochs=10):

    print('enter a gesture name: ')
    name = input()

    pos_data, pos_fs, pos_meta = utils.read_imu_data(name)
    neg_data, neg_fs, neg_meta = utils.read_imu_data('negative')

    pos_x = pos_data['x']
    pos_y = pos_data['y']
    pos_z = pos_data['z']
    neg_x = neg_data['x']
    neg_y = neg_data['y']
    neg_z = neg_data['z']

    # Create time vectors for plotting
    imu_time = np.linspace(0, pos_meta['duration'], len(pos_data))  # Assuming IMU data covers the entire audio duration
    neg_imu_time = np.linspace(0, neg_meta['duration'], len(neg_data))  # Assuming IMU data covers the entire audio duration

    # resample negative data to match fs of positive data if they are slightly off. 
    desired_length = int(round(float(len(neg_data)) /
                               neg_fs * pos_fs))
    neg_data = signal.resample(neg_data, desired_length)


    # Step 3: Plot the data
    if PLOT:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
        # Plot IMU data 
        axs[0].plot(imu_time, pos_x, label='x', color='red')
        axs[0].plot(imu_time, pos_y, label='y', color='green')
        axs[0].plot(imu_time, pos_z, label='z', color='blue')
        axs[0].set_title('IMU Readings')
        axs[0].set_ylabel('Value')
        axs[0].set_xlabel('Time (s)')
        axs[0].legend()

        axs[1].plot(neg_imu_time, neg_x, label='x', color='red')
        axs[1].plot(neg_imu_time, neg_y, label='y', color='green')
        axs[1].plot(neg_imu_time, neg_z, label='z', color='blue')
        axs[1].set_title('IMU Readings')
        axs[1].set_ylabel('Value')
        axs[1].set_xlabel('Time (s)')
        axs[1].legend()

        plt.show()
    
    
    W = 0.5
    overlap = 0.95

    pos_x = utils.continuous_to_windows(pos_x, pos_fs, win_len_sec=W, overlap_frac=overlap)
    pos_y = utils.continuous_to_windows(pos_y, pos_fs, win_len_sec=W, overlap_frac=overlap)
    pos_z = utils.continuous_to_windows(pos_z, pos_fs, win_len_sec=W, overlap_frac=overlap)
    neg_x = utils.continuous_to_windows(neg_x, pos_fs, win_len_sec=W, overlap_frac=overlap)
    neg_y = utils.continuous_to_windows(neg_y, pos_fs, win_len_sec=W, overlap_frac=overlap)
    neg_z = utils.continuous_to_windows(neg_z, pos_fs, win_len_sec=W, overlap_frac=overlap)

    num_examples = pos_x.shape[0]
    num_neg_examples = neg_x.shape[0]
    len_window = pos_x.shape[1]
    print(f'sliced signal to {num_examples} examples. Each window is {len_window} samples for 3 axes')
    print(f'sample per window: {int(W * pos_fs)}')

    flattened_examples = np.empty((num_examples, len_window*3))
    for i in range(num_examples):
        flattened_examples[i, :len_window] = pos_x[i]
        flattened_examples[i, len_window:2*len_window] = pos_y[i]
        flattened_examples[i, 2*len_window:3*len_window] = pos_z[i]

    flattened_neg_examples = np.empty((num_neg_examples, len_window*3))
    for i in range(num_neg_examples):
        flattened_neg_examples[i, :len_window] = neg_x[i]
        flattened_neg_examples[i, len_window:2*len_window] = neg_y[i]
        flattened_neg_examples[i, 2*len_window:3*len_window] = neg_z[i]

    pos_x = flattened_examples
    neg_x = flattened_neg_examples

    # Create labels for each window in the positive and negative dataset.
    pos_labels = np.zeros((pos_x.shape[0], 1))
    pos_labels[:, 0] = 1
    neg_labels = np.zeros((neg_x.shape[0], 1))
    neg_labels[:, 0] = 0

    inputs = np.concatenate((pos_x, neg_x))
    labels = np.concatenate((pos_labels, neg_labels))

    # Shuffle features and labels in place with similar order. 
    features_and_labels = (inputs, labels)
    permutation = np.random.permutation(len(inputs))
    # Shuffle each array in the tuple in-place
    for arr in features_and_labels:
        arr[:] = arr[permutation]
    inputs, labels = features_and_labels
    labels = labels.flatten()

    # Define the model
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    _ = model.fit(inputs, labels, epochs=epochs, batch_size=16)
    return model, inputs, labels
