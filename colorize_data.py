from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from glob import glob
from skimage import io
from skimage.color import gray2rgb

from torchvision.transforms.functional import resize

class ColorizeData(Dataset):
    def __init__(self, image_dir):
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self._data_files = []
        self._load_data(image_dir)

    def _load_data(self, image_dir):
        image_files = glob(f"{image_dir}/*.jpg")

        images = []

        for image_file in image_files:
            images.append(image_file)

        self._data_files = images

        print(f"{len(self._data_files)} files indexed.")
    
    def __len__(self) -> int:
        return len(self._data_files)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = io.imread(self._data_files[index])

        if len(img.shape) < 3 or img.shape[-1] == 1:
            img = gray2rgb(img)

        gray_img = self.input_transform(img/255.0)
        img = self.target_transform(img/255.0)

        return gray_img, img
        