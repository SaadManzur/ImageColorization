import torch

class PSNR:
    def __init__(self, range=255) -> None:
        self._range = range

    def __call__(self, pred, gt):

        batch_size = pred.shape[0]
        channel = pred.shape[1]
        height = pred.shape[2]
        width = pred.shape[3]
        
        mse = torch.mean(torch.mean((pred.view(batch_size, channel, height*width) - gt.view(batch_size, channel, height*width))**2, dim=1), dim=1)

        psnr = 20 * torch.log10(self._range/mse)
        
        return psnr