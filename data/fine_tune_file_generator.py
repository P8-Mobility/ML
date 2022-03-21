import glob
import os
import pathlib
import shutil
import zipfile
import requests

from processing import data_loader


def generate(subject_ids: list[str], api_path: str, api_token: str):
    samples_path: str = str(pathlib.Path().resolve()) + "/data/audio_samples/"

    train_wave_file_lines: list[str] = []
    train_text_file_lines: list[str] = []
    validate_wave_file_lines: list[str] = []
    validate_text_file_lines: list[str] = []

    __retrieve_files_from_api(api_path, api_token, samples_path)

    for file in glob.glob(samples_path + "*.wav"):
        filename: str = file.split('.', 1)[0]
        identifier: str = filename.split('-')[2].split('.')[0]

        wave_entry: str = filename + " " + file
        text_entry: str = filename + " pʰ æː ɐ"

        if identifier in subject_ids:
            validate_wave_file_lines.append(wave_entry)
            validate_wave_file_lines.append('\n')
            validate_text_file_lines.append(text_entry)
            validate_text_file_lines.append('\n')
        else:
            train_wave_file_lines.append(wave_entry)
            train_wave_file_lines.append('\n')
            train_text_file_lines.append(text_entry)
            train_text_file_lines.append('\n')

    validate_wave_file_lines.pop()
    validate_text_file_lines.pop()
    train_wave_file_lines.pop()
    train_text_file_lines.pop()

    __write_lines_to_files_in_dir(str(pathlib.Path().resolve()) + "/data/train/", train_wave_file_lines,
                                  train_text_file_lines)
    __write_lines_to_files_in_dir(str(pathlib.Path().resolve()) + "/data/validate/", validate_wave_file_lines,
                                  validate_text_file_lines)

    loader = data_loader.DataLoader()
    loader.add_folder_to_model('data/audio_samples')
    loader.fit()
    loader.store_processed_files()

    return


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

    with zipfile.ZipFile(temp_zip_file, 'r') as zip_ref:
        zip_ref.extractall(sample_dir)

    os.remove(temp_zip_file)


def __write_lines_to_files_in_dir(directory: str, lines_for_wave: list[str], lines_for_text: list[str]) -> None:
    pathlib.Path(directory + 'wave').touch()
    pathlib.Path(directory + 'text').touch()

    with open(directory + 'wave', "w") as validate_wave_file:
        validate_wave_file.writelines(lines_for_wave)
    with open(directory + 'text', "w") as validate_text_file:
        validate_text_file.writelines(lines_for_text)
