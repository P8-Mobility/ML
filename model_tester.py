import json
import os
import pathlib
import shutil
from pathlib import Path

import numpy

import allosaurus.allosaurus
import data.file_generator
import fine_tune as ft
from allosaurus.allosaurus import app as allo
from data.word_phoneme_map import WordPhonemeMap
from main import __load_config
from processing import data_loader


def recognize(model: str, sample_time_series: numpy.ndarray, sample_sample_rate: int) -> str:
    """
    Predict a sample based on time series and sample rate

    :param model: name of the model to use
    :param sample_time_series: time series representation of the sample
    :param sample_sample_rate: sample rate of the sample
    """
    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/' + model))

    aud = allosaurus.allosaurus.audio.Audio(
        sample_time_series,
        sample_sample_rate)
    return model.recognize(aud)


def recognize_directory(model: str, data_path: str):
    """
    Predict the files found in the data_path, using the model

    :param model: name of the model to use
    :param data_path: path to the samples being predicted
    """
    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/' + model))

    loader = data_loader.DataLoader()
    loader.add_folder_to_model(data_path)
    files = loader.get_data_files()

    for file in files:
        aud = allosaurus.allosaurus.audio.Audio(
            file.time_series,
            file.get_sampling_rate)
        res: str = model.recognize(aud)
        print(file.get_filename + ": " + res)


def get_accuracy(model: str, data_path: str) -> (float, float):
    """
    Predict the files found in the data_path, using the model
    and return the correctly predicted samples / number of samples

    :param model: name of the model to use
    :param data_path: path to the samples being predicted
    """
    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/' + model))

    loader = data_loader.DataLoader()
    loader.add_folder_to_model(data_path)
    files = loader.get_data_files()

    predictions: int = 0
    correct_predictions: int = 0
    incorrect_predictions: dict[str, int] = {}

    for file in files:
        aud = allosaurus.allosaurus.audio.Audio(
            file.time_series,
            file.get_sampling_rate)
        res: str = model.recognize(aud)

        word: str = str(file.get_filename).split('-')[-1].split('.')[0]
        if WordPhonemeMap.contains(word):
            predictions += 1
            if (word == "paere" and WordPhonemeMap.get(word) == res) \
                    or (word != "paere" and WordPhonemeMap.get("paere") != res):
                correct_predictions += 1
            else:
                incorrect_predictions[word] = incorrect_predictions.get(word, 0) + 1

    return correct_predictions, predictions, incorrect_predictions


def run_limited_samples_test():
    """
    Execute test for model fine-tuned with limited samples
    """
    sample_sizes: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]

    for sample_size in sample_sizes:
        __make_subset_sample_folder('data/samples', sample_size)

    with open("result.txt", "w") as result_file:
        result_file.write("")

    for sample_size in sample_sizes:
        model: str = 'paere_' + str(sample_size)
        data.file_generator.generate(str(pathlib.Path().resolve()) + '/data/samples_' + str(sample_size) + '/')
        ft.fine_tune(str(Path().resolve()) + '/data/', model)

        correct_predictions, predictions, incorrect_predictions = get_accuracy(model, 'data/samples_validation')

        with open("result.txt", "a") as result_file:
            if predictions > 0:
                result_file.write(str(sample_size) + ": " + str(correct_predictions) + "/" + str(predictions) +
                                  "(" + str(correct_predictions / predictions) + ") [" +
                                  ", ".join([key + ": " + str(value) for key, value in incorrect_predictions.items()])
                                  + "]\n")


def __make_subset_sample_folder(data_path: str, nr_samples: int):
    new_directory: str = data_path + "_" + str(nr_samples)
    os.makedirs(new_directory, exist_ok=True)

    samples_of_words: dict[str, int] = {}
    for key in WordPhonemeMap.word_phoneme_map.keys():
        samples_of_words[key] = 0

    config = __load_config()
    LOSO_subjects: list[str] = json.loads(config.get('ALLO', 'Subjects'))

    for filename in os.listdir(data_path):
        f = os.path.join(data_path, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.endswith(".wav"):
            word: str = str(filename).split('-')[-1].split('.')[0]
            subject: str = str(filename).split('-')[-2]

            if WordPhonemeMap.contains(word) and samples_of_words.get(word) < nr_samples \
                    and subject not in LOSO_subjects:
                shutil.copy2(f, new_directory + "/" + filename)
                samples_of_words[word] = samples_of_words[word] + 1
