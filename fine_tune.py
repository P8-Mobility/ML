import os
import pathlib
import shutil

import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt


def fine_tune(data_dir: str, model_name: str = "paereModelV2"):
    # Process train data
    pf.prepare_feature(pathlib.Path(data_dir + "/train"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/train"), "uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(pathlib.Path(data_dir + "/validate"), "uni2005/")
    pt.prepare_token(pathlib.Path(data_dir + "/validate"), "uni2005/", 'dan')

    shutil.rmtree(model_name, True)

    # command to fine_tune your data
    os.system("python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=" + model_name + " --path=" + data_dir + " --lang=dan --device_id=-1 --epoch=10")

