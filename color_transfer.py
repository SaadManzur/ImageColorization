import os
import argparse
import numpy as np
from skimage import io
from skimage.color import rgb2lab, lab2rgb

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--out", type=str, default="transfered.jpg")

    return parser.parse_args()

def transfer_color_with_metric(lab_t, mean_s, std_s):
    
    h, w = lab_t.shape[0], lab_t.shape[1]
    
    lab_t = lab_t.reshape(-1, 3)
    
    mean_t = np.mean(lab_t, axis=0)
    std_t = np.std(lab_t, axis=0)
    
    lab = (std_s/std_t)*(lab_t - mean_t) + mean_s
    
    return lab.reshape(h, w, 3)

def transfer_color_profile(lab_s, lab_t):
    
    h, w = lab_t.shape[0], lab_t.shape[1]
    
    lab_s = lab_s[:600, :, :].reshape(-1, 3)
    
    mean_s = np.mean(lab_s, axis=0)
    std_s = np.std(lab_s, axis=0)
    
    return transfer_color_with_metric(lab_t, mean_s, std_s)

if __name__ == "__main__":

    args = get_args()

    assert os.path.exists(args.src) and os.path.exists(args.target)

    src = io.imread(args.src)
    target = io.imread(args.target)

    target_lab = rgb2lab(target/255.)
    src_lab = rgb2lab(src/255.)

    tx_lab = transfer_color_profile(src_lab, target_lab)

    tx_rgb = lab2rgb(tx_lab)*255.0
    tx_rgb = tx_rgb.astype(np.uint8)

    io.imsave(args.out, tx_rgb)