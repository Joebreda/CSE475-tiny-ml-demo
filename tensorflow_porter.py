from everywhereml.code_generators.tensorflow import tf_porter

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import tensorflow_dataset

tf_model, x_train, y_train = tensorflow_dataset.get_model(epochs=50)
#f_model, x_train, y_train = get_model()
# tf_porter() requires:
#   1. the neural network model
#   2. the input data (to detect the input dimensions)
#   3. the output labels (to detect the number of classes - if classification)
#
# Passing `instance_name` will create an instance of the model, so you don't have to
# `area_size` is to control how much memory to allocate for the network
# It is a trial-and-error process
porter = tf_porter(tf_model, x_train, y_train)
cpp_code = porter.to_cpp(instance_name='wiggle_detector', arena_size=4096)

with open('ESP32_embedded_ML_example/wiggle_detector.h', 'w') as f:
    f.write(cpp_code)