import glob
import os
import pathlib
import shutil
import warnings
import zipfile
import requests

from data.word_phoneme_map import WordPhonemeMap
from processing import data_loader


def generate(subject_ids: list[str], api_path: str, api_token: str, should_overwrite: bool = False):
    """
    Fetches audio samples from api and saves samples in their respective directories

    :param subject_ids: IDs of subjects to leave out during training (LOSO)
    :param api_path: path to API endpoint
    :param api_token: Bearer token for API
    :param should_overwrite: whether the directories should be overwritten by new data.
    If False, no new data will be added to non-empty directories
    """
    samples_path: str = str(pathlib.Path().resolve()) + "/data/samples/"

    train_wave_file_lines: list[str] = []
    train_text_file_lines: list[str] = []
    validate_wave_file_lines: list[str] = []
    validate_text_file_lines: list[str] = []

    overwrite_samples: bool = should_overwrite or len(os.listdir(samples_path)) == 0

    if overwrite_samples:
        __retrieve_files_from_api(api_path, api_token, samples_path)

    for file in glob.glob(samples_path + "*.wav"):
        filename: str = file.split('.', 1)[0]
        identifier: str = filename.split('-')[2]
        word: str = filename.split('-')[3]

        if not WordPhonemeMap.contains(word):
            # Word was not found in the word phoneme map, so skip it
            continue

        wave_entry: str = filename + " " + file
        text_entry: str = filename + " " + WordPhonemeMap.get(word)

        if identifier in subject_ids:
            validate_wave_file_lines, validate_text_file_lines = __append_lines(validate_wave_file_lines, wave_entry,
                                                                                validate_text_file_lines, text_entry)
        else:
            train_wave_file_lines, train_text_file_lines = __append_lines(train_wave_file_lines, wave_entry,
                                                                          train_text_file_lines, text_entry)

    if not (validate_wave_file_lines and validate_text_file_lines and train_wave_file_lines and train_text_file_lines):
        return warnings.warn("Unable to fetch test and/or train data")

    # Removing trailing newline from each list
    validate_wave_file_lines.pop()
    validate_text_file_lines.pop()
    train_wave_file_lines.pop()
    train_text_file_lines.pop()

    __write_lines_to_files_in_dir(str(pathlib.Path().resolve()) + "/data/train/", train_wave_file_lines,
                                  train_text_file_lines)
    __write_lines_to_files_in_dir(str(pathlib.Path().resolve()) + "/data/validate/", validate_wave_file_lines,
                                  validate_text_file_lines)

    if overwrite_samples:
        # Only do preprocessing if the files have been overwritten to avoid double preprocessing
        __preprocess_files_and_overwrite(samples_path)


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


def __retrieve_files_from_api(api_path: str, api_token: str, sample_dir: str):
    for filename in os.listdir(sample_dir):
        file_path = os.path.join(sample_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    temp_zip_file = sample_dir + 'temp.zip'
    headers = {"Authorization": "Bearer " + api_token}
    response = requests.get(api_path, headers=headers)
    open(temp_zip_file, 'wb').write(response.content)

    # Unzip files to dir
    with zipfile.ZipFile(temp_zip_file, 'r') as zip_ref:
        zip_ref.extractall(sample_dir)

    os.remove(temp_zip_file)


def __write_lines_to_files_in_dir(directory: str, lines_for_wave: list[str], lines_for_text: list[str]):
    # Ensure that the files exists or create them
    pathlib.Path(directory + 'wave').touch()
    pathlib.Path(directory + 'text').touch()

    with open(directory + 'wave', "w") as wave_file:
        wave_file.writelines(lines_for_wave)
    with open(directory + 'text', "w") as text_file:
        text_file.writelines(lines_for_text)
