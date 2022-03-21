import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt
import configparser
from pathlib import Path
from os import path, getcwd


def fine_tune(train_path: str, validate_path: str):
    # Process train data
    pf.prepare_feature(train_path, "/uni2005/")
    pt.prepare_token(train_path, "/uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(validate_path, "/uni2005/")
    pt.prepare_token(validate_path, "/uni2005/", 'dan')

    # command to fine_tune your data
    #python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=paereModel --path =<pathToData> --lang=dan --device_id=-1 --epoch=10

