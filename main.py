import os
import yaml
import argparse
from train import Trainer
from utils.helper import ConfigStruct

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--ckpt_name", type=str)

    return parser.parse_args()

def load_config(cfg_path):

    with open(cfg_path, 'r') as file_stream:

        cfg_dict = yaml.safe_load(file_stream)
        cfg_obj = ConfigStruct(**cfg_dict)

        file_stream.close()

    return cfg_obj

if __name__ == "__main__":

    args = get_args()

    if args.cfg is None or not os.path.exists(args.cfg):
        raise ValueError("Please provide a config file.")

    cfg = load_config(args.cfg)
    
    trainer = Trainer(cfg)

    if args.evaluate:
        trainer.evaluate(args.ckpt_name, args.out_dir)
    else:
        trainer.train()