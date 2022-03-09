import configparser
import numpy as np
import pickle
import logging

from collections import Counter
from matplotlib import pyplot
from numpy import where
from os import path, getcwd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import OneClassSVM
from audio import Audio
from data_loader import DataLoader


def load_config():
    base_folder = path.dirname(path.abspath(__file__))
    config_file = "config.cnf"
    if not path.exists(config_file) or not path.isfile(config_file):
        print(f"Config file missing... Path should be {getcwd()}/config.cfg")
        return

    config = configparser.ConfigParser()
    config.read(path.join(base_folder, config_file))
    return config


def convert_audio_to_np(audio_files: list[Audio]) -> np.ndarray:
    """
    Converts a list of Audio objects to a numpy array of time series

    :param audio_files: the list of Audio objects
    :return: the converted numpy array
    """
    np_array = np.empty([len(audio_files), audio_files[0].time_series.size])

    for index in range(len(audio_files)):
        np_array[index] = audio_files[index].time_series

    return np_array


def fill_labels(files: list[Audio]) -> np.ndarray:
    """
    Fills numpy array with the correct labels based on the files

    :param files: audio files used to create labels based on
    :return: numpy array with labels
    """
    labels = np.empty(len(files))
    labels.fill(0)
    for index in range(len(files)):
        if files[index].is_wrong:
            labels[index] = 1
    return labels


def filter_wrong(audio_files: list[Audio]) -> list[Audio]:
    files = []
    for file in audio_files:
        if not file.is_wrong:
            files.append(file)

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


class OCC:
    def __init__(self, log: bool):
        self.__log = log
        self.__config = load_config()
        self.__model = self.__get_model()

        logging.basicConfig(
            filename='accuracies.log',
            format='%(asctime)s: %(message)s',
            level=logging.INFO
        )

    def __load_files(self) -> list[Audio]:
        """
        Creates an audio object for each file in the folder

        :return: a list of audio objects
        """
        loader = DataLoader()
        loader.add_folder_to_model(self.__config.get('OCC', 'DataPath'))
        loader.fit()

        audio_files = loader.get_data_files()

        return audio_files

    def __get_model(self, create_new: bool = False) -> OneClassSVM:
        """
        If existing model exists,then it is loaded in. Otherwise a new model is created

        :param create_new: creates new model if True. Otherwise, tries to load model if one exists
        :return: The model used to classify
        """
        filename = self.__config.get('OCC', 'ModelPath', fallback='trained_models/occ_model.sav')

        if self.__log:
            logging.info(f'Name of model: {filename.split("/")[-1]}')

        if path.exists(filename) and not create_new:
            return pickle.load(open(filename, 'rb'))
        else:
            # Define outlier detection model
            return OneClassSVM(gamma='scale', nu=0.01)

    def __split(self, audio_files: list[Audio], subject: str = None) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits the set into two chunks by leaving one subject out

        :param audio_files: the audio files to be split
        :param subject: The subject that contains the other data
        :return: the two train and other sets, as well as their labels
        """
        train_files = []
        test_files = []

        if subject is None:
            subject = audio_files[0].get_id

        for file in audio_files:
            if file.get_id == subject:
                test_files.append(file)
            else:
                train_files.append(file)

        train_files = filter_wrong(train_files)

        if self.__log:
            logging.info(f'Number of train examples: {len(train_files)}. Number of test examples: {len(test_files)}')

        return convert_audio_to_np(train_files), convert_audio_to_np(test_files), fill_labels(train_files), fill_labels(
            test_files)

    def __train(self, train_data: np.ndarray):
        """
        Trains the model based on the training data. Only one label, so there is no need to use the labels

        :param train_data: a numpy array with data the model is trained with
        """
        # Fit on majority class
        self.__model.fit(train_data)
        filename = self.__config.get('OCC', 'ModelPath', fallback='trained_models/occ_model.sav')
        pickle.dump(self.__model, open(filename, 'wb'))

    def predict(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Finds the accuracy of the model based on the other set

        :param test_data: the other data
        :param test_labels: the labels of the other data
        :return: the accuracy
        """
        # Detect outliers in the other set
        # Outputs +1 for normal examples, so-called inliers, and -1 for outliers.
        predicted_labels = self.__model.predict(test_data)

        # Mark inliers 1, outliers -1
        test_labels[test_labels == 1] = -1
        test_labels[test_labels == 0] = 1

        # Calculate score
        accuracy = accuracy_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels, pos_label=1)
        print('Accuracy score: %.3f' % accuracy)
        print('F1 Score: %.3f' % f1)

        if self.__log:
            logging.info(f'Accuracy: {accuracy}\n')

        return accuracy

    def run(self) -> float:
        """
        Default control function that trains on files specified in config and provides accuracy of the model

        :return: the model's accuracy
        """
        files = self.__load_files()

        # Split data set into train and other data
        train_data, test_data, train_labels, test_labels = self.__split(files, '3ElCNtHBZH')

        # Train the model
        self.__train(train_data)

        # Find the accuracy of the model
        return self.predict(test_data, test_labels)


if __name__ == '__main__':
    occ = OCC(False)
    occ.run()
