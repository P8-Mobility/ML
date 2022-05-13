import json
import os
import pathlib
import shutil
from pathlib import Path

import numpy
import matplotlib.pyplot as plt

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
            if word == "paere" and WordPhonemeMap.get(word) == res:
                correct_predictions += 1
            elif word == "myre" and WordPhonemeMap.get(word) == res:
                correct_predictions += 1
            elif word != "paere" and word != "myre" and WordPhonemeMap.get("paere") != res \
                    and WordPhonemeMap.get("myre") != res:
                correct_predictions += 1
            else:
                incorrect_predictions[word] = incorrect_predictions.get(word, 0) + 1

    return correct_predictions, predictions, incorrect_predictions


def run_limited_samples_test():
    """
    Execute test for model fine-tuned with limited samples
    """
    sample_sizes: list[int] = [size for size in range(1, 30)]

    for sample_size in sample_sizes:
        __make_subset_sample_folder('data/samples', sample_size)

    with open("result.txt", "w") as result_file:
        result_file.write("")

    for sample_size in sample_sizes:
        model: str = 'paere_' + str(sample_size)
        data.file_generator.generate(str(pathlib.Path().resolve()) + '/data/samples_' + str(sample_size) + '/',
                                     json.loads(__load_config().get('ALLO', 'Subjects')))
        ft.fine_tune(str(Path().resolve()) + '/data/', model)

        correct_predictions, predictions, incorrect_predictions = get_accuracy(model, 'data/samples_validation')

        with open("result.txt", "a") as result_file:
            if predictions > 0:
                result_file.write(str(sample_size) + ": " + str(correct_predictions) + "/" + str(predictions) +
                                  " (" + str(correct_predictions / predictions) + ") [" +
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

            if WordPhonemeMap.contains(word):
                if subject not in LOSO_subjects and samples_of_words.get(word) < nr_samples:
                    samples_of_words[word] = samples_of_words[word] + 1
                    shutil.copy2(f, new_directory + "/" + filename)
                elif subject in LOSO_subjects:
                    shutil.copy2(f, new_directory + "/" + filename)


def plot_result():
    """
    Plots the results found in 'results.txt' and saves the figure in the root of the project
    """
    if os.path.isfile("result.txt"):
        test_results: dict[int, list[int]] = {0: [0, 0]}

        with open("result.txt", "r") as f:
            for line in f:
                line_segments: list[str] = line.split(" ")
                test_results[int(line_segments[0].split(':')[0])] = [int(value) for value in line_segments[1].split('/')]

        # set title and axis labels
        plt.title("Results from fine-tuning with subset")
        plt.xlabel('Subset size (samples of each word)')
        plt.ylabel('Accuracy (%)')

        # set size of the graph
        plt.axis([0, max([subset_size for subset_size in test_results.keys()]), 0, 100])

        # make sure that ticks are added to the x-axis for every other subset size
        plt.xticks([res - res % 2 for res in test_results.keys()])

        # make sure that ticks are added to the y-axis for every 10 percent
        yticks = [y - y % 10 for y in range(0, 100)]
        yticks.append(100)
        plt.yticks(yticks)

        # plot results
        plt.plot([subset_size for subset_size in test_results.keys()], [(sub_results[0] / sub_results[1] * 100 if sub_results[1] != 0 else 0) for sub_results in test_results.values()], marker ='.')

        # add dotted line representing our 70% target accuracy
        plt.plot([70 for _ in test_results.keys()], 'r--')
        plt.savefig("plottedSubsetFineTuningResults.png")

    else:
        print("Unable to generate plot, as result.txt does not exists")
