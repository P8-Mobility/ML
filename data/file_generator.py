import glob
import math
import os
import pathlib
import shutil
import warnings
import zipfile
import requests

from data.word_phoneme_map import WordPhonemeMap
from processing import data_loader


def generate(samples_path: str = "/data/samples/", LOSO_subjects=None):
    """
    Fetches audio samples from api and saves samples in their respective directories

    :param samples_path: path to the folder containing the samples that should be used to generate training files
    """
    if LOSO_subjects is None:
        LOSO_subjects = []

    file_paths: list[str] = glob.glob(samples_path + "*.wav")
    training_set: list[str] = []
    validation_set: list[str] = []

    for fp in file_paths:
        subject = str(fp).split('-')[-2]
        if subject in LOSO_subjects:
            validation_set.append(fp)
        else:
            training_set.append(fp)

    # generate wave and text files for training
    train_wave_file_lines, train_text_file_lines = __get_lines_for_wave_and_text_files(training_set)

    if not (train_wave_file_lines and train_text_file_lines):
        return warnings.warn("Unable to generate training wave and text files")

    __write_lines_to_files_in_dir(str(pathlib.Path().resolve()) + "/data/train/", train_wave_file_lines,
                                  train_text_file_lines)

    # generate wave and text files for validation
    validate_wave_file_lines, validate_text_file_lines = __get_lines_for_wave_and_text_files(validation_set)

    if not (validate_wave_file_lines and validate_text_file_lines):
        return warnings.warn("Unable to generate validation wave and text files")

    __write_lines_to_files_in_dir(str(pathlib.Path().resolve()) + "/data/validate/", validate_wave_file_lines,
                                  validate_text_file_lines)


def retrieve_files_from_api(subject_ids: list[str], api_path: str, api_token: str, samples_dir: str):
    """
    Retrieves .zip file from the REST API, extracts the samples,
    and splits them into samples_dir and 'samples_validation'

    :param subject_ids: identifiers for subjects to be moved into 'samples_validation' directory
    :param api_path: endpoint of the API
    :param api_token: bearer token for the API
    :param samples_dir: directory to store samples with identifier not in subject_ids
    :return:
    """
    # clear the target sample directory
    __clear_directory(samples_dir)

    # download samples as .zip file
    temp_zip_file = samples_dir + 'temp.zip'
    headers = {"Authorization": "Bearer " + api_token}
    response = requests.get(api_path, headers=headers)
    open(temp_zip_file, 'wb').write(response.content)

    # Unzip files to dir
    with zipfile.ZipFile(temp_zip_file, 'r') as zip_ref:
        zip_ref.extractall(samples_dir)
    os.remove(temp_zip_file)
    __preprocess_files_and_overwrite(samples_dir)


def __get_lines_for_wave_and_text_files(files: list[str]):
    wave_file_lines: list[str] = []
    text_file_lines: list[str] = []

    for file in files:
        filename: str = file.split('.', 1)[0]
        word: str = filename.split('-')[3]

        if not WordPhonemeMap.contains(word):
            continue

        wave_entry: str = filename + " " + file
        text_entry: str = filename + " " + WordPhonemeMap.get(word)

        wave_file_lines, text_file_lines = __append_lines(wave_file_lines, wave_entry,
                                                          text_file_lines, text_entry)

    if len(wave_file_lines) > 0 and len(text_file_lines) > 0:
        # remove trailing newlines
        wave_file_lines.pop()
        text_file_lines.pop()

    return wave_file_lines, text_file_lines


def __append_lines(wave_file_lines: list[str], wave_entry: str, text_file_lines: list[str], text_entry: str) -> tuple[
    list[str], list[str]]:
    wave_file_lines.append(wave_entry)
    wave_file_lines.append('\n')
    text_file_lines.append(text_entry)
    text_file_lines.append('\n')

    return wave_file_lines, text_file_lines


def __preprocess_files_and_overwrite(samples_path: str):
    loader = data_loader.DataLoader()
    loader.add_folder_to_model(samples_path)
    loader.fit()
    loader.store_processed_files(samples_path)


def __write_lines_to_files_in_dir(directory: str, lines_for_wave: list[str], lines_for_text: list[str]):
    # Ensure that the files exists or create them
    pathlib.Path(directory + 'wave').touch()
    pathlib.Path(directory + 'text').touch()

    with open(directory + 'wave', "w") as wave_file:
        wave_file.writelines(lines_for_wave)
    with open(directory + 'text', "w") as text_file:
        text_file.writelines(lines_for_text)


def __clear_directory(directory):
    """
    Clears the directory or creates it, if it does not exist

    :param directory: directory to clear or create
    """
    os.makedirs(directory, exist_ok=True)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Due to %s' % (file_path, e))
