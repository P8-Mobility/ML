import os
import pathlib

import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt


def fine_tune(data_dir: str):
    # Process train data
    pf.prepare_feature(pathlib.Path(data_dir + "/train"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/train"), "uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(pathlib.Path(data_dir + "/validate"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/validate"), "uni2005/", 'dan')


    # command to fine_tune your data
    os.system("python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=paereModelV2 --path=" + data_dir + " --lang=dan --device_id=-1 --epoch=10")

