import configparser
import glob
import os
import pathlib
import shutil

import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt
import allosaurus.allosaurus.app as allo
import allosaurus.allosaurus.audio as audio

from pathlib import Path

from data.word_phoneme_map import WordPhonemeMap
from processing import data_loader


def fine_tune(data_dir: str, config: configparser.ConfigParser, model_name: str = "paere"):
    """
    Generate a new model that is fine-tuned based on the uni2005 model with the data located in data_dir

    :param data_dir: path to directory containing the samples to perform fine-tuning with (should contain train and validate directories)
    :param config: configuration file
    :param model_name: the name of the new model
    """

    result_map = {}

    # Allosaurus training directory path
    allo_path = "allosaurus/allosaurus/pretrained/"

    # Process train data
    pf.prepare_feature(pathlib.Path(data_dir + "/train"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/train"), "uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(pathlib.Path(data_dir + "/validate"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/validate"), "uni2005/", 'dan')

    for epochs in range(70, 5, -5):
        print("Training model with " + str(epochs) + " epochs...")

        model = model_name + "_" + str(epochs)

        # Ignore errors is set to true, to avoid exception if the folder does not exist
        shutil.rmtree(pathlib.Path(allo_path + model), ignore_errors=True)

        # command to fine_tune your data
        os.system("python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=" + model + " --path=" + data_dir + " --lang=dan --device_id=0 --epoch=" + str(epochs))

        # Find all saved states of the model during training for deletion
        file_list = glob.glob(allo_path + model_name + "_" + str(epochs)+"/model_*.pt")
        print("Housekeeping: removing "+str(len(file_list))+" intermediate models from training...")

        for filePath in file_list:
            try:
                os.remove(filePath)
            except:
                print("Error while deleting file : ", filePath)

        total_classified, correct_classified = recognize_directory(model, config)
        result_map[model] = (correct_classified/total_classified)

    for key, value in result_map.items():
        print(key + " -> " + str(value))


def recognize_directory(model: str, config: configparser.ConfigParser) -> (int, int):
    """
    Predict the files found in the data_path, using the model

    :param model: name of the model to use
    :param config: configuration file
    :return: number of total classified recordings and number of correctly classified recordings
    """
    model = allo.read_recognizer(alt_model_path=Path('allosaurus/allosaurus/pretrained/' + model))
    phoneme_map = WordPhonemeMap

    loader = data_loader.DataLoader()
    loader.add_folder_to_model("data/samples_validation")
    files = loader.get_data_files()

    correct_classified = 0
    total_classified = 0

    subjects = config.get('ALLO', 'Subjects')

    for file in files:
        if file.get_id not in subjects:
            continue

        word = file.get_word
        phoneme = phoneme_map.get(word)

        aud = audio.Audio(file.time_series, file.get_sampling_rate)
        res: str = model.recognize(aud)

        # if phoneme == res:
        if (word == "paere" and phoneme == res) or (word != "paere" and res != phoneme_map.get("paere")):
            correct_classified += 1

        if phoneme != "":
            total_classified += 1

        # print(file.get_filename + ": " + res)

    # print("Correct classified: " + str(correct_classified))
    # print("Total classified: " + str(total_classified))
    # print("Accuracy: " + str(correct_classified/total_classified))

    return total_classified, correct_classified
