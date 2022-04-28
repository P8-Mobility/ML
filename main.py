import configparser
import json
import os
import pathlib
from typing import Union

import numpy
import allosaurus.allosaurus.app as allo
import allosaurus.allosaurus.audio as audio
import data.file_generator
import fine_tune as ft
from pathlib import Path
from processing import data_loader
from data.word_phoneme_map import WordPhonemeMap


def main():
    model: str = 'paere'
    config = __load_config()
    data.file_generator.generate(json.loads(config.get('ALLO', 'Subjects')), config.get('ALLO', 'API_Path'),
                                  config.get('ALLO', 'API_Token'), True)
    ft.fine_tune(str(pathlib.Path().resolve()) + '/data/', config, model)

    for epochs in range(70, 5, -5):
        recognize_directory("paere_" + str(epochs), 'data/test', config)

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


def recognize_directory(model: str, path: str, config: configparser.ConfigParser):
    """
    Predict the files found in the data_path, using the model

    :param path:
    :param model: name of the model to use
    :param config: configuration file
    :return: number of total classified recordings and number of correctly classified recordings
    """
    model_name = model

    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/' + model))
    phoneme_map = WordPhonemeMap

    loader = data_loader.DataLoader()
    loader.add_folder_to_model(path)
    files = loader.get_data_files()

    correct_classified = 0
    total_classified = 0

    for file in files:
        word = file.get_word
        phoneme = phoneme_map.get(word)

        aud = audio.Audio(file.time_series, file.get_sampling_rate)
        res: str = model.recognize(aud)

        if (word == "paere" and phoneme == res) or (word != "paere" and res != phoneme_map.get("paere")):
            correct_classified += 1

        if phoneme != "":
            total_classified += 1

        print(file.get_filename + ": " + res)

    # print("Correct classified: " + str(correct_classified))
    # print("Total classified: " + str(total_classified))
    print("### "+model_name+" ###")
    print("Accuracy: " + str(correct_classified/total_classified))


def recognize(model: str, sample_time_series: numpy.ndarray, sample_sample_rate: int) -> str:
    """
    Predict a sample based on time series and sample rate

    :param model: name of the model to use
    :param sample_time_series: time series representation of the sample
    :param sample_sample_rate: sample rate of the sample
    """
    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/' + model))

    aud = audio.Audio(
        sample_time_series,
        sample_sample_rate)
    return model.recognize(aud)


if __name__ == "__main__":
    main()
