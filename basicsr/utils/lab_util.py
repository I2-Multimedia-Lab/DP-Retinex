import cv2
import torch
import numpy as np


class RGBLABConverter:
    """
    A utility class for converting RGB images to LAB space and back using OpenCV.

    Attributes:
        return_l_only (bool): If True, only the L channel will be returned in RGB to LAB conversion.
        return_ab_only (bool): If True, only the A and B channels will be returned in RGB to LAB conversion.
    """

    def __init__(self, return_l_only=False, return_ab_only=False):
        self.return_l_only = return_l_only
        self.return_ab_only = return_ab_only

    def rgb_to_lab(self, image_tensor):
        """
        Convert RGB image tensor to LAB space. Allows selective return of L, A, B channels.

        Args:
            image_tensor (torch.Tensor): Input tensor of shape (B, 3, H, W) with values in [0, 1].

        Returns:
            torch.Tensor: LAB image tensor of shape (B, 3, H, W) with values in [0, 1]
                          if return_l_only or return_ab_only is False, otherwise (B, 1, H, W) or (B, 2, H, W).
        """
        # Scale and convert to numpy format, rearrange for OpenCV compatibility (B, H, W, C)
        image_np = image_tensor.cpu().numpy()
        image_np = image_np.transpose(0, 2, 3, 1)  # Convert to (B, H, W, C)

        # Perform RGB to LAB conversion for each batch
        lab_images = [cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in image_np]
        lab_images = np.stack(lab_images, axis=0)  # (B, H, W, C)

        # Convert to tensor and normalize channels
        lab_tensor = torch.from_numpy(lab_images).float().to(image_tensor.device)
        # Return selective channels if specified
        if self.return_l_only:
            return lab_tensor[:, :, :, 0].unsqueeze(1)  # Return L channel only with shape (B, 1, H, W)
        elif self.return_ab_only:
            return lab_tensor[:, :, :, 1:].permute(0, 3, 1, 2)  # Return A and B channels with shape (B, 2, H, W)
        else:
            return lab_tensor.permute(0, 3, 1, 2)  # Return full LAB tensor with shape (B, 3, H, W) L|A|B

    def lab_to_rgb(self, lab_tensor):
        """
        Convert LAB image tensor back to RGB space.

        Args:
            lab_tensor (torch.Tensor): LAB tensor of shape (B, 3, H, W) or selective channels in LAB with values [0, 1].

        Returns:
            torch.Tensor: RGB image tensor of shape (B, 3, H, W) with values in [0, 1].
        """
        # Ensure LAB tensor is in (B, H, W, C) format and denormalize to OpenCV range
        lab_tensor = lab_tensor.permute(0, 2, 3, 1)  # Convert to (B, H, W, C)
        lab_np = lab_tensor.cpu().numpy()

        # Convert each LAB image to RGB
        rgb_images = [cv2.cvtColor(img.astype(np.float32), cv2.COLOR_LAB2RGB) for img in lab_np]
        rgb_images = np.stack(rgb_images, axis=0)  # (B, H, W, C)

        # Convert to tensor and normalize back to [0, 1]
        rgb_tensor = torch.from_numpy(rgb_images).float().to(lab_tensor.device)
        return rgb_tensor.permute(0, 3, 1, 2)  # Return in (B, 3, H, W) format
