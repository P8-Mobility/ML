import allosaurus.allosaurus.bin.prep_feat as pf
import allosaurus.allosaurus.bin.prep_token as pt
import configparser
from pathlib import Path
from os import path, getcwd


def load_config():
    base_folder = path.dirname(path.abspath(__file__))
    config_file = "config.cnf"
    if not path.exists(config_file) or not path.isfile(config_file):
        print(f"Config file missing... Path should be {getcwd()}/config.cfg")
        return

    config = configparser.ConfigParser()
    config.read(path.join(base_folder, config_file))
    return config


def fine_tune():
    config = load_config()
    fp_train = Path(config.get('allo', 'DataPath') + 'train')
    fp_validate = Path(config.get('allo', 'DataPath') + 'validate')

    # Process train data
    pf.prepare_feature(fp_train, "/uni2005/")
    pt.prepare_token(fp_train, "/uni2005/", 'dan')

    # Process validate data
    pf.prepare_feature(fp_validate, "/uni2005/")
    pt.prepare_token(fp_validate, "/uni2005/", 'dan')

    # command to fine_tune your data
    #python -m allosaurus.allosaurus.bin.adapt_model --pretrained_model=uni2005 --new_model=paereModel --path =<pathToData> --lang=dan --device_id=-1 --epoch=10

