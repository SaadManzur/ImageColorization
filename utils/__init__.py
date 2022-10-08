from torch import Tensor
import torchvision.transforms as T

from .transforms import *

basic_rgb_input_transformation = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
basic_rgb_target_transformation = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
basic_lab_transformation = T.Compose([ResizeImage(), NormalizeLab(), ToTensor()])
binned_transformation = T.Compose([ResizeImage(), DiscretizeImage(), NormalizeLChannel(), TensorizeBins()])