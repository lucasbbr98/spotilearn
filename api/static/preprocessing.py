import numpy as np


def train_test_split(x_data, y_data, percantage=0.8, shuffle=False):
    num_training_indices = int(percantage * len(x_data))
    x_train, x_test = x_data[:num_training_indices], x_data[num_training_indices:]
    y_train, y_test = y_data[:num_training_indices], y_data[num_training_indices:]
    return x_train, x_test, y_train, y_test

def shuffle_array(arr):
    np.random.shuffle(arr)
    return arr

def add_ones(matrix):
    ones = np.ones((matrix.shape[0], 1))
    return np.c_[ones, matrix]

def standard_features(x):
    x -= np.mean(x)  # the -= means can be read as x = x- np.mean(x)
    x /= np.std(x)  # the /= means can be read as x = x/np.std(x)
    return x
