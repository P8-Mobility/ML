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

from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM


def split(set: np.ndarray, percentage: int) -> [np.ndarray, np.ndarray]:
    """
    Splits the set into two chunks with sizes based on the percentage specified

    :param set: the array to be split
    :param percentage: the percentage of the set that the size of the first chunk is
    :return: the two sets
    """
    split_value = len(set) / 100 * percentage
    return set[:split_value], set[split_value:]


# Define dataset to consist of 10.000 examples with 10 in the minority class and 9.990 in the majority class
data, labels = make_classification(n_samples=10000, n_features=2, n_redundant=0, n_clusters_per_class=1,
                                   weights=[0.999], flip_y=0, random_state=4)
# Summarize class distribution
counter = Counter(labels)
print(counter)

# # Scatter plot of examples by class label
# for label, _ in counter.items():
#     row_ix = where(labels == label)[0]
#     pyplot.scatter(data[row_ix, 0], data[row_ix, 1], label=str(label))
# pyplot.legend()
# pyplot.show()

# Define outlier detection model
model = OneClassSVM(gamma='scale', nu=0.01)

# Split data set into train and test data
train_data, test_data = split(data, 70)
train_labels, test_labels = split(labels, 70)

# Fit on majority class
train_data = train_data[train_labels == 0]
model.fit(train_data)

# Detect outliers in the test set
# Outputs +1 for normal examples, so-called inliers, and -1 for outliers.
predicted_labels = model.predict(test_data)

# Mark inliers 1, outliers -1
test_labels[test_labels == 1] = -1
test_labels[test_labels == 0] = 1

# Calculate score
score = f1_score(test_labels, predicted_labels, pos_label=-1)
print('F1 Score: %.3f' % score)

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
