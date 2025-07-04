import os
import random
from torch.utils import data
from typing import List, Dict, Tuple
import basicsr.data.util as util
import torch

class Dataset_SIDImage(data.Dataset):
    """
    Dataset for SIDImage, which loads paired Low Quality (LQ) and Ground Truth (GT) images from
    the specified directories for training and testing.

    Args:
        opt (dict): A dictionary containing various options for dataset configuration,
                    such as paths, image sizes, and training parameters.
    """

    def __init__(self, opt: Dict[str, any]):
        super().__init__()

        self.opt = opt
        self.cache_data = opt.get('cache_data', False)
        self.half_N_frames = opt.get('N_frames', 1) // 2
        self.GT_root = opt['dataroot_gt']
        self.LQ_root = opt['dataroot_lq']
        self.io_backend_opt = opt['io_backend']
        self.data_type = opt['io_backend']
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}

        # Validate IO backend type
        if self.data_type == 'lmdb':
            raise ValueError('LMDB is not supported during validation/test.')

        # Cache data if required
        self.imgs_LQ, self.imgs_GT = {}, {}

        # Gather all subfolders
        subfolders_LQ, subfolders_GT = self._get_subfolders()

        # Process each pair of LQ and GT subfolders
        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
            self._process_subfolder_pair(subfolder_LQ, subfolder_GT)

    def _get_subfolders(self) -> Tuple[List[str], List[str]]:
        """
        Retrieve the subfolders from the LQ and GT roots depending on the phase (train/test).

        Returns:
            Tuple: A tuple containing two lists - LQ and GT subfolders.
        """
        subfolders_LQ_origin = util.glob_file_list(self.LQ_root)
        subfolders_GT_origin = util.glob_file_list(self.GT_root)

        subfolders_LQ, subfolders_GT = [], []
        if self.opt['phase'] == 'train':
            for lq_folder, gt_folder in zip(subfolders_LQ_origin, subfolders_GT_origin):
                name = os.path.basename(lq_folder)
                if '0' in name[0] or '2' in name[0]:
                    subfolders_LQ.append(lq_folder)
                    subfolders_GT.append(gt_folder)
        else:
            for lq_folder, gt_folder in zip(subfolders_LQ_origin, subfolders_GT_origin):
                name = os.path.basename(lq_folder)
                if '1' in name[0]:
                    subfolders_LQ.append(lq_folder)
                    subfolders_GT.append(gt_folder)

        return subfolders_LQ, subfolders_GT

    def _process_subfolder_pair(self, subfolder_LQ: str, subfolder_GT: str):
        """
        Process the LQ and GT image paths in a given subfolder pair (LQ, GT) and
        update the data info dictionary with relevant data.

        Args:
            subfolder_LQ (str): Path to the LQ subfolder.
            subfolder_GT (str): Path to the GT subfolder.
        """
        subfolder_name = os.path.basename(subfolder_LQ)
        img_paths_LQ = util.glob_file_list(subfolder_LQ)
        img_paths_GT = util.glob_file_list(subfolder_GT)

        max_idx = len(img_paths_LQ)
        self.data_info['path_LQ'].extend(img_paths_LQ)
        self.data_info['path_GT'].extend(img_paths_GT)
        self.data_info['folder'].extend([subfolder_name] * max_idx)

        # Generate index and border information
        for i in range(max_idx):
            self.data_info['idx'].append(f"{i}/{max_idx}")

        border_l = [0] * max_idx
        for i in range(self.half_N_frames):
            border_l[i] = 1
            border_l[max_idx - i - 1] = 1
        self.data_info['border'].extend(border_l)

        # Cache image paths if necessary
        if self.cache_data:
            self.imgs_LQ[subfolder_name] = img_paths_LQ
            self.imgs_GT[subfolder_name] = img_paths_GT

    def __getitem__(self, index: int) -> Dict[str, any]:
        """
        Retrieve the LQ and GT images corresponding to a given index.

        Args:
            index (int): The index of the image pair in the dataset.

        Returns:
            dict: A dictionary containing the LQ and GT images and related information.
        """
        folder = self.data_info['folder'][index]
        #print(index)
        idx, max_idx = map(int, self.data_info['idx'][index].split('/'))
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx]
        img_GT_path = self.imgs_GT[folder][0]

        # Read image sequences
        img_LQ = util.read_img_seq2([img_LQ_path], self.opt['train_size'])[0]
        img_GT = util.read_img_seq2([img_GT_path], self.opt['train_size'])[0]

        if self.opt['phase'] == 'train':
            img_LQ, img_GT = self._augment_and_crop(img_LQ, img_GT)

        return {
            'lq': img_LQ,
            'gt': img_GT,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path,
            'gt_path': img_GT_path
        }

    def _augment_and_crop(self, img_LQ: torch.Tensor, img_GT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform data augmentation and crop the images to the specified size.

        Args:
            img_LQ (torch.Tensor): Low-quality image tensor.
            img_GT (torch.Tensor): Ground-truth image tensor.

        Returns:
            Tuple: Augmented and cropped LQ and GT images.
        """
        LQ_size = self.opt['LQ_size']
        GT_size = self.opt['GT_size']

        if LQ_size != GT_size:
            print("Warning: LQ_size must be equal to GT_size.")

        _, H, W = img_GT.shape
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))

        img_LQ = img_LQ[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]
        img_GT = img_GT[:, rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size]

        img_LQ_l = [img_LQ]
        img_LQ_l.append(img_GT)

        # Augmentation (flip and rotate)
        augmented = util.augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
        return augmented[0], augmented[1]

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data_info['path_LQ'])