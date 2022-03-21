import os

import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt
import configparser
from pathlib import Path
from os import path, getcwd


def fine_tune(data_dir: str):
    # Process train data
    pf.prepare_feature(data_dir + "/train", "/uni2005/")
    pt.prepare_token(data_dir + "/train", "/uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(data_dir + "/validate", "/uni2005/")
    pt.prepare_token(data_dir + "/validate", "/uni2005/", 'dan')

    # command to fine_tune your data
    os.system("python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=paereModel"
              "--path=" + data_dir + " --lang=dan --device_id=-1 --epoch=10")

