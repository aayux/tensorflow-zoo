# -*- coding: utf-8 -*-

""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
"""

__author__ = "Aayush Yadav"
__email__ = "aayushyadav96@gmail.com"

import os
import pickle as pckl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


def fetch_batch(path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(path + "/data_batch_" + str(batch_id), mode="rb") as file:
        batch = pckl.load(file, encoding="latin1")

    features = batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch["labels"]

    return features, labels

def process_and_pickle(features, labels, filename):
    """
    Preprocess data and save it to file
    """
    # Normalize a list of sample image data in the range of 0 to 1
    features = features/255
    
    # One hot encode a list of sample labels
    one_hot_labels = np.zeros((len(labels), 10))
    one_hot_labels[list(np.indices((len(labels),))) + [labels]] = 1
    labels = one_hot_labels

    if filename == "./data/train.pckl" and os.path.exists(filename):
        with open(filename, "rb") as train_dump:
            pckld_features, pckld_labels = pckl.load(train_dump)
        features = np.concatenate((features, pckld_features))
        labels = np.concatenate((labels, pckld_labels))
    
    pckl.dump((features, labels), open(filename, "wb"))


def process_cifar10(path):
    """
    Preprocess the training data
    """
    n_batches = 5

    for batch_i in range(1, n_batches + 1):
        features, labels = fetch_batch(path, batch_i)

        # Preprocess and save a batch of training data
        process_and_pickle(
            features,
            labels,
            "./data/train.pckl")

    with open(path + "/test_batch", mode="rb") as file:
        batch = pckl.load(file, encoding="latin1")

    # load the testing data
    test_features = batch["data"].reshape((len(batch["data"]), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch["labels"]

    # Preprocess and Save all testing data
    process_and_pickle(
        np.array(test_features),
        np.array(test_labels),
        "./data/test.pckl")

def load_cifar10(filename):
    """
    Load the Preprocessed data
    """
    features, labels = pckl.load(open(filename, mode="rb"))
    return features, labels

def split_data(x, y, val_split):
    """
    Split into train/validation set
    """
    idx = -1 * int(val_split * float(len(y)))
    x_train, x_val = x[:idx], x[idx:]
    y_train, y_val = y[:idx], y[idx:]

    print("Train/Val split: {:d}/{:d}".format(len(y_train), len(y_val)))

    return x_train, y_train, x_val, y_val

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    print ("Generating batch iterator ...")
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]