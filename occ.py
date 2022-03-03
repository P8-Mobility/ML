import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import csv
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where

from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import OneClassSVM

import preprocessing


def show_distribution(data: np.ndarray, labels: np.ndarray):
    # Summarize class distribution
    counter = Counter(labels)
    print(counter)

    # Scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(labels == label)[0]
        pyplot.scatter(data[row_ix, 0], data[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()


def split(set: np.ndarray, percentage: int) -> [np.ndarray, np.ndarray]:
    """
    Splits the set into two chunks with sizes based on the percentage specified

    :param set: the array to be split
    :param percentage: the percentage of the set that the size of the first chunk is
    :return: the two sets
    """
    split_value = int(len(set) / 100 * percentage)
    return set[:split_value], set[split_value:]


def predict(model: OneClassSVM, test_data: np.ndarray, test_labels: np.ndarray) -> float:
    """
    Finds the accuracy of the model based on the test set

    :param model: the model that is being tested
    :param test_data: the test data
    :param test_labels: the labels of the test data
    :return: the accuracy
    """
    # Detect outliers in the test set
    # Outputs +1 for normal examples, so-called inliers, and -1 for outliers.
    predicted_labels = model.predict(test_data)

    # Mark inliers 1, outliers -1
    test_labels[test_labels == 1] = -1
    test_labels[test_labels == 0] = 1

    # Calculate score
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels, pos_label=1)
    print('Accuracy score: %.3f' % accuracy)
    print('F1 Score: %.3f' % f1)
    return accuracy


def run():
    path = "OCCFiles"
    files = preprocessing.preprocess_sound(path)
    data = np.empty([len(files), files[0].size])
    for index in range(len(files)):
        data[index] = files[index]
    labels = np.empty(len(files))
    labels.fill(0)

    # show_distribution(data, labels)

    # Define outlier detection model
    model = OneClassSVM(gamma='scale', nu=0.01)

    # Split data set into train and test data
    train_data, test_data = split(data, 70)
    train_labels, test_labels = split(labels, 70)

    # Fit on majority class
    model.fit(train_data)

    # Find the accuracy of the model
    accuracy = predict(model, test_data, test_labels)

# def run():
#     data = extract_and_normalize()
#     classes = extract_class_names()
#     convert_to_pandas(data, classes)
#
#
# def extract_and_normalize():
#     file_paths = librosa.util.find_files("OCCFiles/", ext=['wav'])
#     files = np.empty([6, 501, 809])
#     for path in file_paths:
#         time_series, sampling_rate = librosa.load(path, sr=48000)  # Makes floating point time series
#         files += librosa.feature.mfcc(time_series, sampling_rate)
#     return librosa.util.normalize(files)
#
#
# def extract_class_names():
#     classes = []
#     with open('class.csv', 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             if len(row) >= 2 and row[1] not in classes:
#                 classes.append(row[1])
#     return classes
#
#
# def convert_to_pandas(data: np.ndarray, classes: list):
#     data_frame = {'data': data, 'label': classes[0]}


if __name__ == '__main__':
    run()
