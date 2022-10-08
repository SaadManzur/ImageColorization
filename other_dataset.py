from logging.handlers import BufferingHandler
import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage import io
from skimage.color import gray2rgb
from utils import basic_lab_transformation, binned_transformation

class ImageDataset(Dataset):

    def __init__(self, image_dir) -> None:
        super().__init__()

        self._data_files = []

        self.input_transform = basic_lab_transformation

        self._load_data(image_dir)

    def _load_data(self, image_dir):
        image_files = glob(f"{image_dir}/*.jpg")

        images = []

        for image_file in image_files:
            images.append(image_file)

        self._data_files = images

        print(f"{len(self._data_files)} files indexed.")

    def __getitem__(self, index):

        img = io.imread(self._data_files[index])

        if len(img.shape) < 3 or img.shape[-1] == 1:
            img = gray2rgb(img)

        lab_img = self.input_transform(img/255.0)

        return lab_img[:1, :, :], lab_img[1:, :, :]

    def __len__(self):
        return len(self._data_files)

class DiscreteImageDataset(ImageDataset):

    def __init__(self, image_dir, train=True) -> None:
        super().__init__(image_dir, train)

        self.input_transform = binned_transformation

    def __getitem__(self, index):
        
        img = io.imread(self._data_files[index])

        if len(img.shape) < 3 or img.shape[-1] == 1:
            img = gray2rgb(img)

        L, binned = self.input_transform(img/255.0)

        return L, binned