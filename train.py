import os
import time
import cv2
import lpips
import numpy as np
from tqdm import tqdm
from skimage.color import lab2rgb, rgb2lab
from skimage.io import imsave
from pytorch_msssim import ssim

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from colorize_data import ColorizeData
from other_dataset import ImageDataset, DiscreteImageDataset
from models.basic_model import Net
from models.binned_model import BinnedNet
from utils.helper import Progress, AverageTracker
from utils.transforms import UnnormalizeLab, UnnormalizeImageTensor
from utils.metrics import PSNR

class Trainer:
    def __init__(self, cfg):
        self._train_dir = cfg.TRAIN_DIR
        self._val_dir = cfg.VAL_DIR
        self._exp_base = cfg.EXP_BASE_DIR
        self._exp_name = cfg.EXP_NAME
        self._model_name = cfg.MODEL
        self._lr = float(cfg.LEARNING_RATE)
        self._batch_size = int(cfg.BATCH_SIZE)
        self._epochs = int(cfg.EPOCHS)
        self._save_every = int(cfg.SAVE_EVERY_EPOCH)
        self._resume_from = cfg.RESUME_FROM
        self._dataset = cfg.DATASET
        self._loss = cfg.LOSS
        self._mode = cfg.MODE
        self._channel = cfg.CHANNEL

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    def save_checkpoint(self, model, optim, loss, epoch, suffix=None):

        current_exp_path = os.path.join(self._exp_base, self._exp_name)

        if not os.path.exists(current_exp_path):
            os.makedirs(current_exp_path)

        ckpt_name = f"ckpt_{epoch:05}.pth.tar" if suffix is None else f"ckpt_{suffix}.pth.tar"

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "loss": loss
        }, os.path.join(current_exp_path, ckpt_name))

        print(f"Checkpoint: {ckpt_name}")

    def load_checkpoint(self, model, optim, ckpt_name):

        current_exp_path = os.path.join(self._exp_base, self._exp_name)

        ckpt = torch.load(os.path.join(current_exp_path, ckpt_name))

        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])

        print(f"Loaded checkpoint: {os.path.join(current_exp_path, ckpt_name)}")

        return model, optim, ckpt["loss"], ckpt["epoch"]

    def model_criterion_dataset(self):

        model, criterion, train_dataset, val_dataset = None, None, None, None

        if self._model_name == "basic":
            model = Net(mode=self._mode, out_channel=self._channel).to(self._device)

            if self._loss == "LPIPS":
                print("Using LPIPS Loss")
                criterion = lpips.LPIPS(net='vgg').to(self._device)
            elif self._loss == "L2":
                print("Using L2 Loss")
                criterion = nn.MSELoss().to(self._device)

            if self._dataset == "basic":
                train_dataset = ColorizeData(self._train_dir)
                val_dataset = ColorizeData(self._val_dir)
            else:
                print("Using LAB dataset")
                train_dataset = ImageDataset(self._train_dir)
                val_dataset = ImageDataset(self._val_dir)
        elif self._model_name == "binned":
            model = BinnedNet().to(self._device)
            criterion = nn.CrossEntropyLoss().to(self._device)
            train_dataset = DiscreteImageDataset(self._train_dir)
            val_dataset = DiscreteImageDataset(self._val_dir, train=False)
        else:
            raise ValueError("Model name not supported.")

        return model, criterion, train_dataset, val_dataset

    def inverse_transform_image(self, img):

        if self._dataset == "basic":
            img = UnnormalizeImageTensor((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img.detach().cpu())
            img = img.numpy()
            img *= 255
            img = img.astype(np.uint8)

            return img
        elif self._dataset == "lab":
            return img
        else:
            return img

    def get_lab_lpips(self, img, pred, target, lpips_):
        pred_ab = pred[0].mul(254.0).add(-128.0).detach().cpu().numpy()
        target_ab = target[0].mul(254.0).add(-128.0).detach().cpu().numpy()
        img_l = img[0].mul(100.0).detach().cpu().numpy()

        pred_img = lab2rgb(np.vstack((img_l, pred_ab)).transpose(1, 2, 0)).astype(np.float32)
        target_img = lab2rgb(np.vstack((img_l, target_ab)).transpose(1, 2, 0)).astype(np.float32)

        pred_img = torch.from_numpy(pred_img)[None, :, :, :].to(self._device)
        target_img = torch.from_numpy(target_img)[None, :, :, :].to(self._device)

        mean_and_std = torch.tensor([0.5, 0.5, 0.5]).to(self._device)
        pred_img = torch.sub(pred_img, mean_and_std).div(mean_and_std)
        target_img = torch.sub(target_img, mean_and_std).div(mean_and_std)

        assert torch.min(target_img) >= -1

        pred_img = pred_img.permute(0, 3, 1, 2)
        target_img = target_img.permute(0, 3, 1, 2)

        return lpips_(pred_img, target_img), pred_img, target_img

    def get_lab_psnr(self, img, pred, target, psnr):

        pred_ab = pred[0].mul(254.0).add(-128.0).detach().cpu().numpy()
        target_ab = target[0].mul(254.0).add(-128.0).detach().cpu().numpy()
        img_l = img[0].mul(100.0).detach().cpu().numpy()

        pred_img = lab2rgb(np.vstack((img_l, pred_ab)).transpose(1, 2, 0)).astype(np.float32)
        target_img = lab2rgb(np.vstack((img_l, target_ab)).transpose(1, 2, 0)).astype(np.float32)

        pred_img = torch.from_numpy(pred_img)[None, :, :, :].to(self._device)
        target_img = torch.from_numpy(target_img)[None, :, :, :].to(self._device)

        mean_and_std = torch.tensor([0.5, 0.5, 0.5]).to(self._device)
        pred_img = torch.sub(pred_img, mean_and_std).div(mean_and_std)
        target_img = torch.sub(target_img, mean_and_std).div(mean_and_std)

        assert torch.min(target_img) >= -1

        return psnr(pred_img.permute(0, 3, 1, 2), target_img.permute(0, 3, 1, 2))

    def post_process(self, gray, pred):

        gray_lab = rgb2lab(gray/255.)
        pred_lab = rgb2lab(pred/255.)

        pred = lab2rgb(np.dstack((gray_lab[:, :, :1], pred_lab[:, :, 1:])))
        pred = (pred*255).astype(np.uint8)

        return pred

    def train(self):
        
        #Pick model, criterion and dataset based on config file
        model, criterion, train_dataset, val_dataset = self.model_criterion_dataset()

        optimizer = torch.optim.Adam(params=model.parameters(), lr=self._lr)

        train_dataloader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # load checkpoint if any
        if len(self._resume_from) > 0:
            model, optimizer, loss, i_epoch = self.load_checkpoint(model, optimizer, self._resume_from)

        # train loop

        num_batches = len(train_dataloader)

        max_psnr = 0
        min_lpips = np.inf

        for i_epoch in range(self._epochs):

            progress = Progress(f"Train Epoch {i_epoch}/{self._epochs}")
            loss_tracker = AverageTracker()
            load_time_tracker = AverageTracker()
            batch_time_tracker = AverageTracker()

            now = time.time()

            model.train()

            print(f"Epoch {i_epoch+1}/{self._epochs}")

            for i_batch, (img, target) in enumerate(train_dataloader):

                data_load_time = time.time()-now

                img, target = img.to(self._device).float(), target.to(self._device).float()

                optimizer.zero_grad()
                pred = model(img)

                loss = criterion(target, pred)
                loss_tracker.update(loss.mean().item())

                batch_time = time.time() - now

                loss.mean().backward()
                optimizer.step()

                load_time_tracker.update(data_load_time)
                batch_time_tracker.update(batch_time)

                progress.update(f"Batch: {i_batch}/{num_batches} | Data Time: {load_time_tracker.avg:.2f}s | Batch Time: {batch_time_tracker.avg:.2f}s | Loss: {loss_tracker.avg:.3f}")

                now = time.time()

            progress.close()

            _, lpips_current = self.validate(model, val_dataloader)

            if lpips_current < min_lpips:
                min_lpips = lpips_current
                self.save_checkpoint(model, optimizer, loss, i_epoch, "best")

            if not (i_epoch+1)%self._save_every:
                self.save_checkpoint(model, optimizer, loss, i_epoch)

    def validate(self, model, dataloader):

        model.eval()

        psnr = PSNR(2) #-1 to 1
        psnr_tracker = AverageTracker()
        lpips_tracker = AverageTracker()
        ssim_tracker = AverageTracker()

        lpips_ = lpips.LPIPS(net='alex').to(self._device)

        with torch.no_grad():

            progress = Progress("Validation")

            for i_batch, (img, target) in enumerate(dataloader):
                img, target = img.to(self._device).float(), target.to(self._device).float()

                pred = model(img)

                if self._dataset == "lab":
                    lpips_metric, _, _ = self.get_lab_lpips(img, pred, target, lpips_)
                    lpips_tracker.update(lpips_metric.mean().item())
                    psnr_metric = self.get_lab_psnr(img, pred, target, psnr)
                    psnr_tracker.update(psnr_metric.item())
                else:
                    psnr_metric = psnr(pred, target)
                    psnr_tracker.update(psnr_metric.item())
                    lpips_metric = lpips_(pred, target)
                    lpips_tracker.update(lpips_metric.mean().item())

                #mean_and_std = torch.tensor([0.5, 0.5, 0.5]).to(self._device)
                #pred_un = torch.mul(pred.permute(0, 2, 3, 1), mean_and_std).add(mean_and_std)*255.0
                #target_un = torch.mul(target.permute(0, 2, 3, 1), mean_and_std).add(mean_and_std)*255.0

                #ssim_ = ssim(pred_un.permute(0, 3, 1, 2), target_un.permute(0, 3, 1, 2), data_range=255, size_average=True)
                #ssim_tracker.update(ssim_.item())

                progress.update(f"Batch: {i_batch}/{len(dataloader)} | PSNR: {psnr_tracker.avg:.2f} | LPIPS: {lpips_tracker.avg:.2f}")

            progress.close()

        return psnr_tracker.avg, lpips_tracker.avg

    def evaluate(self, ckpt_name, out_dir):

        model, _, _, dataset = self.model_criterion_dataset()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self._lr)

        model, optimizer, _, _ = self.load_checkpoint(model, optimizer, ckpt_name)

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        model.eval()

        psnr = PSNR(2) #-1 to 1
        psnr_tracker = AverageTracker()
        lpips_tracker = AverageTracker()
        ssim_tracker = AverageTracker()

        lpips_ = lpips.LPIPS(net='alex').to(self._device)

        with torch.no_grad():

            progress = Progress("Validation")

            for i_batch, (img, target) in enumerate(dataloader):
                img, target = img.to(self._device).float(), target.to(self._device).float()

                pred = model(img)

                if self._dataset == "lab":
                    psnr_metric = self.get_lab_psnr(img, pred, target, psnr)
                    psnr_tracker.update(psnr_metric.item())
                    lpips_metric, pred, target = self.get_lab_lpips(img, pred, target, lpips_)
                    lpips_tracker.update(lpips_metric.mean().item())
                else:
                    psnr_metric = psnr(pred, target)
                    psnr_tracker.update(psnr_metric.item())
                    lpips_metric = lpips_(pred, target)
                    lpips_tracker.update(lpips_metric.mean().item())

                progress.update(f"Batch: {i_batch}/{len(dataloader)} | PSNR: {psnr_tracker.avg:.2f} | LPIPS: {lpips_tracker.avg:.2f}")

                if out_dir is not None and os.path.exists(out_dir) and os.path.isdir(out_dir):
                    #if self._dataset != "lab":
                    
                    pred = self.inverse_transform_image(pred[0])
                    gray = self.inverse_transform_image(torch.tile(img[0], (3, 1, 1)))
                    target = self.inverse_transform_image(target[0])
                    
                    pred = self.post_process(gray, pred)

                    imsave(os.path.join(out_dir, f"{i_batch}.jpg"), np.hstack((gray, pred, target)))

            progress.close()

        return psnr_tracker.avg, lpips_tracker.avg