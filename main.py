import configparser
import json
import os
import pathlib

import allosaurus.allosaurus.app as allo
import allosaurus.allosaurus.audio
import data.fine_tune_file_generator
import fine_tune as ft
from pathlib import Path
from processing import data_loader


def main():
    config = __load_config()
    data.fine_tune_file_generator.generate(json.loads(config.get('allo', 'Subjects')), config.get('allo', 'API_Path'),
                                           config.get('allo', 'API_Token'))
    ft.fine_tune(str(pathlib.Path().resolve()) + "/data/")
    recognize()

    return


def __load_config():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = "config.cnf"
    if not os.path.exists(config_file) or not os.path.isfile(config_file):
        print(f"Config file missing... Path should be {os.getcwd()}/config.cfg")
        return

    config = configparser.ConfigParser()
    config.read(os.path.join(base_folder, config_file))
    return config


def recognize():
    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/paereModelV2'))

    loader = data_loader.DataLoader()
    # loader.change_setting("scale_length", False)
    loader.add_folder_to_model('data/audio_samples')
    # loader.fit()
    # loader.store_processed_files()
    files = loader.get_data_files()

    for file in files:
        aud = allosaurus.allosaurus.audio.Audio(
            file.time_series,
            file.get_sampling_rate)
        res: str = model.recognize(aud)
        print(file.get_filename + ": " + res)


if __name__ == "__main__":
    main()

# run inference -> æ l u s ɔ ɹ s
# print("pære: " + )
# print("bære: " + model.recognize('bre1.wav'))
# print("avg: " + model.recognize('avg_sound.wav'))
