import numpy as np
import torch
import torchvision.transforms as T
from skimage.transform import resize
from skimage.color import rgb2lab

class NormalizeLab(object):
    
    def __init__(self) -> None:
        pass

    def __call__(self, image):

        assert len(image.shape) == 3

        lab = rgb2lab(image)

        lab[:, :, :1] /= 100.0
        lab[:, :, 1:] = (lab[:, :, 1:] + 128.0) / 256.0

        return lab

class UnnormalizeImageTensor(object):

    def __init__(self, mean, std) -> None:
        self._mean = torch.from_numpy(np.asarray(mean))
        self._std = torch.from_numpy(np.asarray(std))

    def __call__(self, img_tensor):
        img_tensor = img_tensor.permute(1, 2, 0)

        img_tensor = img_tensor.mul(self._std).add(self._mean)

        return img_tensor

class UnnormalizeLab(object):

    def __init__(self) -> None:
        pass

    def __call__(self, L, ab):

        L = L.numpy()*100.0
        ab = (ab.numpy() * 254.0) - 128.0

        return L, ab

class ToTensor(object):

    def __init__(self) -> None:
        pass

    def __call__(self, image):

        assert image.shape == (256, 256, 3)

        image = torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))

        return image

class ResizeImage(object):

    def __init__(self, size=256) -> None:
        self.size = size

    def __call__(self, image):
        
        assert image.shape[-1] == 3

        image = resize(image, (self.size, self.size))

        return image

class DiscretizeImage(object):

    def __init__(self, bin_path="priors/pts_in_hull.npy"):
        self._bin_centers = np.load(bin_path).T
    
    def __call__(self, image):

        assert len(image.shape) == 3

        lab = rgb2lab(image)
        L, ab = lab[:, :, :1], lab[:, :, 1:]
        
        distances = np.sqrt(((ab[:, :, np.newaxis, :]-self._bin_centers.T[np.newaxis, np.newaxis, :, :])**2).sum(axis=3))
        
        binned = np.argmin(distances, axis=2)
        
        return (lab[:, :, :1], binned)
    
class NormalizeLChannel(object):
    
    def __init__(self):
        pass
    
    def __call__(self, L_and_binned):
        L, binned = L_and_binned
        
        assert L.shape[-1] == 1
        
        L /= 100.0
        
        return L, binned
        
class GenerateABComponent(object):
    
    def __init__(self, bin_path="priors/pts_in_hull.npy"):
        self._bin_centers = np.load(bin_path).T
        pass
    
    def __call__(self, tensor):
        
        ab_image = np.zeros((tensor.shape[0], tensor.shape[1], 2))
        
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                ab_image[i, j] = self._bin_centers[:, tensor[i, j]]
        
        return ab_image


class TensorizeBins(object):

    def __init__(self) -> None:
        pass

    def __call__(self, L_and_binned):
        
        L, binned = L_and_binned
        
        assert L.shape[-1] == 1

        L_tensor = torch.from_numpy(L.astype(np.float32).transpose(2, 0, 1))
        binned_tensor = torch.nn.functional.one_hot(torch.from_numpy(binned), num_classes=313).permute(2, 0, 1)

        return L_tensor, binned_tensor