import configparser
import json
import os
import pathlib
import numpy
import allosaurus.allosaurus.app as allo
import allosaurus.allosaurus.audio
import data.file_generator
import fine_tune as ft
from pathlib import Path
from processing import data_loader


def main():
    model: str = 'paere'
    config = __load_config()
    data.fine_tune_file_generator.generate(json.loads(config.get('ALLO', 'Subjects')), config.get('ALLO', 'API_Path'),
                                           config.get('ALLO', 'API_Token'))
    ft.fine_tune(str(pathlib.Path().resolve()) + '/data/', model)
    recognize(model, 'data/validation_samples')

    return


def __load_config():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = 'config.cnf'
    if not os.path.exists(config_file) or not os.path.isfile(config_file):
        print(f'Config file missing... Path should be {os.getcwd()}/config.cfg')
        return

    config = configparser.ConfigParser()
    config.read(os.path.join(base_folder, config_file))
    return config


def recognize(model: str, data_path: str):
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


if __name__ == "__main__":
    main()
