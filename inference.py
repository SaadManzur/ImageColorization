import os
from sqlite3 import OperationalError
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage import io
from skimage.color import gray2rgb, rgb2lab, lab2rgb

import torch

from train import Trainer
from utils import binned_transformation
from utils import basic_lab_transformation
from utils import basic_rgb_input_transformation
from utils import basic_rgb_target_transformation
from utils.transforms import UnnormalizeImageTensor
from utils.helper import ConfigStruct
from models.basic_model import Net
from models.binned_model import BinnedNet

def load_config(cfg_path):

    with open(cfg_path, 'r') as file_stream:

        cfg_dict = yaml.safe_load(file_stream)
        cfg_obj = ConfigStruct(**cfg_dict)

        file_stream.close()

    return cfg_obj

class Inference(Trainer):

    def __init__(self, cfg, ckpt_name, postprocess=True, output_original=False):
        super().__init__(cfg)

        self._ckpt_name = ckpt_name
        self._in_transform = None
        self._target_transform = None
        self._post_process = postprocess
        self._output_original = output_original

        self._model = self.load_model()
        self._model.eval()

    def load_model(self):

        if self._dataset == "basic":
            # Need RGB Image
            model = Net().to(self._device)
            optimizer = torch.optim.Adam(model.parameters())
            model, _, _, _ = self.load_checkpoint(model, optimizer, self._ckpt_name)
            self._in_transform = basic_rgb_input_transformation
            self._out_transform = basic_rgb_target_transformation
            
            return model

        elif self._dataset == "lab":
            model = Net(out_channel=2).to(self._device)
            optimizer = torch.optim.Adam(model.parameters())
            model, _, _, _ = self.load_checkpoint(model, optimizer, self._ckpt_name)
            self._in_transform = basic_lab_transformation

            return model

        else:
            model = BinnedNet().to(self._device)
            optimizer = torch.optim.Adam(model.parameters())
            model, _, _, _ = self._load_checkpoint(model, optimizer, self._ckpt_name)
            self._in_transform = binned_transformation

            return model

    def transform_image(self, img):

        if self._dataset == "basic":
            img = self._in_transform(img)

            return img

        elif self._dataset == "lab":

            print("Inference not scripted for lab model yet.")
            exit(0)

            img = self._in_transform(img/255.)

            return img[:1, :, :]

        else:
            print("Inference not scripted for binned model yet.")
            exit(0)

            return img

    def infer_single_image(self, img_path, out_path):

        assert os.path.exists(img_path)
        assert os.path.exists(out_path)

        img_name, img_ext = os.path.splitext(os.path.basename(img_path))

        img = io.imread(img_path)

        if len(img.shape) < 3 or img.shape[-1] == 1:
            img = gray2rgb(img)

        img = self.transform_image(img).to(self._device)

        pred = self._model(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))[0]

        pred = self.inverse_transform_image(pred)
        gray = self.inverse_transform_image(torch.tile(img, (3, 1, 1)))

        if self._post_process:
            pred = self.post_process(gray, pred)

        if self._output_original:
            io.imsave(os.path.join(out_path, f"{img_name}_gray{img_ext}"), gray)
        
        io.imsave(os.path.join(out_path, f"{img_name}_colored{img_ext}"), pred)

    def infer_from_directory(self, img_dir, out_path, ext):

        if not os.path.exists(out_path):
            try:
                os.makedirs(out_path)
            except:
                raise OperationalError()

        images = glob(os.path.join(img_dir, f"*.{ext}"))

        for i, image in tqdm(enumerate(images)):
            self.infer_single_image(image, out_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, help="You can specify a single image or directory")
    parser.add_argument("--out", type=str, help="Output directory only")
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--convert", action="store_true", help="If you provide an RGB image, convert to grayscale first")
    parser.add_argument("--ckpt_name", type=str, default="ckpt_best.pth.tar")
    parser.add_argument("--ext", type=str, default="jpg")
    parser.add_argument("--disable_postprocess", action='store_true', default=False, help="Will show the raw model output without post processing")
    parser.add_argument("--output_original", action='store_true', default=False, help="If you want to save the original output in the same directory for comparison.")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()

    if args.cfg is None or not os.path.exists(args.cfg):
        raise ValueError("Please provide a config file.")

    if not args.ckpt_name:
        raise ValueError("Please specify a checkpoint name.")

    cfg_file = load_config(args.cfg)

    inference = Inference(cfg_file, args.ckpt_name, postprocess=not args.disable_postprocess, output_original=args.output_original)

    if os.path.isdir(args.inp):
        inference.infer_from_directory(args.inp, args.out, ext=args.ext)
    else:
        inference.infer_single_image(args.inp, args.out)