import configparser
import os
import pathlib
from typing import Union

from pathlib import Path
import json

import fine_tune as ft
import data.file_generator
import model_tester


def main():
    model: str = "paere"

    # fetch data from rest
    config = __load_config()
    if (not os.path.isdir(str(pathlib.Path().resolve()) + '/data/samples/')) or len(
            os.listdir(str(pathlib.Path().resolve()) + '/data/samples/')) == 0:
        data.file_generator.retrieve_files_from_api(json.loads(config.get('ALLO', 'Subjects')),
                                                    config.get('ALLO', 'API_Path'), config.get('ALLO', 'API_Token'),
                                                    str(pathlib.Path().resolve()) + '/data/samples/')

    data.file_generator.generate(str(pathlib.Path().resolve()) + '/data/samples/',
                                 json.loads(config.get('ALLO', 'Subjects')))
    model_tester.run_limited_samples_test()
    # model_tester.plot_result()
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


if __name__ == "__main__":
    main()
