import configparser
import os
import pathlib
import shutil
from typing import Union

import numpy
import allosaurus.allosaurus.app as allo
import allosaurus.allosaurus.audio
from pathlib import Path
import json

import fine_tune as ft
import data.file_generator
from data.word_phoneme_map import WordPhonemeMap
from processing import data_loader


def main():
    # fetch data if samples do not exist
    samples_dir = str(pathlib.Path().resolve()) + '/data/samples/'
    config = __load_config()
    if not (os.path.exists(samples_dir) and len(os.listdir(samples_dir)) > 0):
        data.file_generator.retrieve_files_from_api(json.loads(config.get('ALLO', 'Subjects')),
                                                    config.get('ALLO', 'API_Path'), config.get('ALLO', 'API_Token'),
                                                    str(pathlib.Path().resolve()) + '/data/samples/')

    run_limited_samples_test()
    return


def __load_config() -> Union[configparser.ConfigParser, None]:
    base_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = 'config.cnf'
    if not os.path.exists(config_file) or not os.path.isfile(config_file):
        print(f'Config file missing... Path should be {os.getcwd()}/config.cfg')
        return

    config = configparser.ConfigParser()
    config.read(os.path.join(base_folder, config_file))
    return config


def fine_tune_model(model_name: str = "paere"):
    """
    Run fine-tuning on specified model

    :param model_name: name of the model to fine-tune
    """
    data.file_generator.generate(str(pathlib.Path().resolve()) + '/data/samples/')
    ft.fine_tune(str(Path().resolve()) + '/data/', model_name)


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


def run_limited_samples_test():
    """
    Execute test for model fine-tuned with limited samples
    """
    sample_sizes: list[int] = [10, 20, 30, 40, 50]

    for sample_size in sample_sizes:
        __make_subset_sample_folder('data/samples', sample_size)

    with open("result.txt", "w") as result_file:
        result_file.write("")

    for sample_size in sample_sizes:
        model: str = 'paere_' + str(sample_size)
        data.file_generator.generate(str(pathlib.Path().resolve()) + '/data/samples_' + str(sample_size) + '/')
        ft.fine_tune(str(Path().resolve()) + '/data/', model)

        correct_predictions, predictions = __get_accuracy(model, 'data/samples_validation')

        with open("result.txt", "a") as result_file:
            print(str(correct_predictions) + ' ' + str(predictions))
            if predictions > 0:
                result_file.write(str(sample_size) + ": " + str(correct_predictions) + "/" + str(predictions) +
                                  "(" + str(correct_predictions / predictions) + ")\n")


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


def __get_accuracy(model: str, data_path: str) -> (float, float):
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
            print(word + " is not recognized as a valid word")

    return correct_predictions, predictions


if __name__ == "__main__":
    main()
