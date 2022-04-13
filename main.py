import configparser
import os
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


def fine_tune_model(model_name: str = "paere", refresh_data: bool = False):
    config = __load_config()
    data.file_generator.generate(json.loads(config.get('ALLO', 'Subjects')), config.get('ALLO', 'API_Path'),
                                 config.get('ALLO', 'API_Token'), refresh_data)
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
    # loader.fit() ToDo preprocess here when we start using the model for recognizing
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
    sample_sizes: list[int] = [10, 20, 30, 40, 50]

    for sample_size in sample_sizes:
        model: str = 'paere_' + str(sample_size)
        config = __load_config()
        data.file_generator.generate(json.loads(config.get('ALLO', 'Subjects')), config.get('ALLO', 'API_Path'),
                                     config.get('ALLO', 'API_Token'), False)
        ft.fine_tune(str(Path().resolve()) + '/data/', model)
        __make_subset_sample_folder('data/samples', sample_size)
        print(__get_accuracy(model, 'data/samples_' + str(sample_size)))


def __make_subset_sample_folder(data_path: str, nr_samples: int):
    loader = data_loader.DataLoader()
    loader.add_folder_to_model(data_path)
    files = loader.get_data_files()

    new_directory: str = data_path + "_" + str(nr_samples)
    os.makedirs(new_directory, exist_ok=True)

    samples_of_words: dict[str, int] = {}
    for key in WordPhonemeMap.word_phoneme_map.keys():
        samples_of_words[key] = 0

    for file in files:
        word: str = str(file.get_filename).split('-')[-1].split('.')[0]
        if WordPhonemeMap.contains(word) and samples_of_words.get(word) < nr_samples:
            file.save(new_directory + "/" + file.get_filename)
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
            if WordPhonemeMap.get(word) == res:
                correct_predictions += 1
        else:
            print(word + " is not recognized as a valid word")

    return correct_predictions, predictions


if __name__ == "__main__":
    main()
