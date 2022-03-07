import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import csv
import pickle
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
from os import path, listdir
import audio
import transformer

from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import OneClassSVM

import preprocessing


def load_files() -> (list[audio.Audio], float):
    """
    Creates an audio object for each file in the folder

    :return: a list of audio objects
    """
    path = "OCCFiles/"
    files = []
    duration_list = []

    for file in listdir(path):
        file = audio.load(path + file)
        files.append(file)
        duration_list.append(file.get_duration())

    avg_duration = sum(duration_list) / len(duration_list)

    return files, avg_duration


def preprocess_files(files: list[audio.Audio], avg_duration: float) -> list[audio.Audio]:
    for file in files:
        transformer.normalize(file)
        transformer.remove_noice(file)
        transformer.trim(file)
        transformer.stretch_to_same_time(file, avg_duration)

    return files


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


def get_model(create_new: bool = False) -> OneClassSVM:
    """
    If existing model exists,then it is loaded in. Otherwise a new model is created

    :param create_new: creates new model if True. Otherwise, tries to load model if one exists
    :return: The model used to classify
    """
    filename = 'trained_models/occ_model.sav'
    if path.exists(filename) and not create_new:
        return pickle.load(open(filename, 'rb'))
    else:
        # Define outlier detection model
        return OneClassSVM(gamma='scale', nu=0.01)


def split(audio_files: list[audio.Audio], subject: str = None) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the set into two chunks by leaving one subject out

    :param audio_files: the audio files to be split
    :param subject: The subject that contains the test data
    :return: the two train and test sets, as well as their labels
    """
    train_files = []
    test_files = []

    if subject is None:
        subject = audio_files[0].get_id()

    for file in audio_files:
        if file.get_id() == subject:
            test_files.append(file)
        else:
            train_files.append(file)

    train_labels = np.empty(len(train_files))
    test_labels = np.empty(len(test_files))
    train_labels.fill(0)
    test_labels.fill(0)

    return convert_audio_to_np(train_files), convert_audio_to_np(test_files), train_labels, test_labels


def convert_audio_to_np(audio_files: list[audio.Audio]) -> np.ndarray:
    """
    Converts a list of Audio objects to a numpy array of time series

    :param audio_files: the list of Audio objects
    :return: the converted numpy array
    """
    np_array = np.empty([len(audio_files), audio_files[0].time_series.size])

    for index in range(len(audio_files)):
        np_array[index] = audio_files[index].time_series

    return np_array


def train(model: OneClassSVM, train_data: np.ndarray) -> OneClassSVM:
    """
    Trains the model based on the training data. Only one label, so there is no need to use the labels

    :param model: the model to be trained
    :param train_data: a numpy array with data the model is trained with
    :return: the trained model
    """
    # Fit on majority class
    model.fit(train_data)
    filename = 'trained_models/occ_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model


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
    files, avg_duration = load_files()
    files = preprocess_files(files, avg_duration)

    # Split data set into train and test data
    train_data, test_data, train_labels, test_labels = split(files)

    # show_distribution(data, labels)

    # Create or load a model
    model = get_model()

    # Train the model
    train(model, train_data)

    # Find the accuracy of the model
    accuracy = predict(model, test_data, test_labels)


if __name__ == '__main__':
    run()
