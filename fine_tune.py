import os
import pathlib
import shutil

import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt


def fine_tune(data_dir: str, model_name: str = "paere"):
    """
    Generate a new model that is fine-tuned based on the uni2005 model with the data located in data_dir

    :param data_dir: path to directory containing the samples to perform fine-tuning with (should contain train and validate directories)
    :param model_name: the name of the new model
    """
    # Process train data
    pf.prepare_feature(pathlib.Path(data_dir + "/train"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/train"), "uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(pathlib.Path(data_dir + "/validate"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/validate"), "uni2005/", 'dan')

    # Ignore errors is set to true, to avoid exception if the folder does not exist
    shutil.rmtree(pathlib.Path("allosaurus/allosaurus/pretrained/" + model_name), ignore_errors=True)

    # command to fine_tune your data
    os.system("python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=" + model_name + " --path=" + data_dir + " --lang=dan --device_id=0 --epoch=50")
