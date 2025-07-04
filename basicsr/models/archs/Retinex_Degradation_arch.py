import torch
import torch.nn as nn
from basicsr.models.archs.Transformer import Transformer
from basicsr.models.archs.diffusion import diffusion
from basicsr.models.archs.Encoder import EfficientIlluminationDecoder, denoise
from basicsr.models.archs.Img_Restoration import LYT

class Retinex_Degradationv2(nn.Module):
    def __init__(self):
        super(Retinex_Degradationv2, self).__init__()
        timesteps = 4
        linear_start = 0.1
        linear_end = 0.99

        self.LYT = LYT().cuda()
        self.encoder = EfficientIlluminationDecoder(in_channels=7, hidden_dim=512).cuda()
        self.condition = EfficientIlluminationDecoder(in_channels=4, hidden_dim=512).cuda()
        self.denoise = denoise(hidden_dim=512, feats=64, timesteps=4).cuda()
        self.diffusion = diffusion.DDPM(denoise=self.denoise,
                                        condition=self.condition, feats=64, timesteps=timesteps,
                                        linear_start=linear_start, linear_end=linear_end).cuda()
        # Target this module for visualization
        self.Restorer = Transformer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 4]).cuda()
        # defult num_blocks=[1, 2, 4]

    def _rgb_to_ycbcr(self, image):
        # image: (batch_size, 3, height, width) tensor, with values in range [0, 1]
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 0.5
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 0.5
        yuv = torch.stack((y, u, v), dim=1)
        return yuv

    def _ycbcr_to_rgb(self, yuv):
        # yuv: (batch_size, 3, height, width) tensor, with values in range [0, 1]
        y, u, v = yuv[:, 0, :, :], yuv[:, 1, :, :], yuv[:, 2, :, :]

        # Inverse transformation (YCbCr to RGB)
        u = u - 0.5  # shift U channel
        v = v - 0.5  # shift V channel

        r = y + 1.13983 * v
        g = y - 0.39465 * u - 0.58060 * v
        b = y + 2.03211 * u
        rgb = torch.stack((r, g, b), dim=1)
        # Clamp to valid range before returning
        return torch.clamp(rgb, 0, 1)

    def forward(self, x):
        # Handle both training (list/tuple) and inference (single tensor) input
        if isinstance(x, (list, tuple)):
            img = x[0]
            gt = x[1] if len(x) > 1 else None  # Handle cases where GT might not be provided
            training_mode = True if gt is not None else False  # Simple check for training mode
        else:
            img = x
            gt = None
            training_mode = False

        b, *_ = img.shape
        # Ensure input image is on the correct device
        img = img.cuda()
        if gt is not None:
            gt = gt.cuda()

        process_img = self.LYT(img)

        # Convert to YCbCr
        img_ycbcr = self._rgb_to_ycbcr(img)
        gt_ycbcr = self._rgb_to_ycbcr(gt) if training_mode and gt is not None else None

        combine_process_img = torch.cat((img_ycbcr, process_img), dim=1)

        if training_mode and gt_ycbcr is not None:
            combine_gt_img = torch.cat((img_ycbcr, process_img, gt_ycbcr), dim=1)
            cdp = self.encoder(combine_gt_img)  # cdp:B*C C:512
            cdp_diff = self.diffusion(combine_process_img, cdp)  # cdp_diff:B*C C:512

            # Pass YCbCr image to Restorer
            image_restored_ycbcr = self.Restorer(img_ycbcr, process_img, cdp_diff)
            # Can compress dynamic range of values/ensure smooth gradients during training
            image_restored_ycbcr = torch.sigmoid(image_restored_ycbcr)
            enhancement_image = self._ycbcr_to_rgb(image_restored_ycbcr)
            return {
                'enhancement_image': enhancement_image,
                "process_img": process_img,
                "process_gt_img": gt_ycbcr[:, 0, :, :].unsqueeze(1) if gt_ycbcr is not None else None,
                'cdp': cdp,
                'cdp_diff': cdp_diff,
            }
        else:
            # Inference mode
            cdp_diff = self.diffusion(combine_process_img, x_0=None)  # Generate condition
            # Pass YCbCr image to Restorer
            image_restored_ycbcr = self.Restorer(img_ycbcr, process_img, cdp_diff)
            image_restored_ycbcr = torch.sigmoid(image_restored_ycbcr)
            enhancement_image = self._ycbcr_to_rgb(image_restored_ycbcr)
            return enhancement_image

