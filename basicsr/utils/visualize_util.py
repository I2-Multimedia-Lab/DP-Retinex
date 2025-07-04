import os
import numpy as np
import torch
from PIL import Image

# 全局计数器，用于生成唯一文件名
_SAVE_TENSOR_COUNTER = 0

def save_tensor(tensor, prefix="tensor", npy_dir="./saved_npy", image_dir="./saved_images"):
    """
    保存张量为 .npy 文件，支持多通道和多图像保存，使用全局计数器生成唯一文件名。
    
    Args:
        tensor (torch.Tensor): 要保存的张量 (B,C,H,W 格式)
        prefix (str): 文件名前缀，默认为 "tensor"
        npy_dir (str): 保存 .npy 文件的路径
        image_dir (str): 保存图像的路径
    
    Returns:
        dict: 包含保存的文件路径信息
    """
    global _SAVE_TENSOR_COUNTER
    
    # 确保保存路径存在
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # 确保张量在 CPU 上，并转换为 NumPy 数组
    np_array = tensor.cpu().detach().numpy()

    # 生成唯一的文件名
    unique_filename = f"{prefix}_{_SAVE_TENSOR_COUNTER}"
    npy_filename = f"{unique_filename}.npy"
    
    # 保存 .npy 文件
    npy_path = os.path.join(npy_dir, npy_filename)
    np.save(npy_path, np_array)
    print(f"Tensor saved to {npy_path}")

    # 存储文件路径信息
    saved_files = {
        "npy_path": npy_path,
        "image_paths": []
    }

    # 处理图像保存 (针对 B,C,H,W 格式)
    if tensor.dim() == 4:
        # 遍历 batch 中的每个图像
        for b in range(np_array.shape[0]):
            np_data = np_array[b]

            # 处理不同通道数的情况
            if np_data.shape[0] == 1:  # 单通道灰度图
                # 压缩通道维度，转换为 H,W
                np_data_2d = np_data.squeeze()
                
                # 归一化和缩放到 0-255
                np_data_2d = ((np_data_2d - np_data_2d.min()) / 
                              (np_data_2d.max() - np_data_2d.min()) * 255).astype(np.uint8)
                
                # 保存为PNG
                image_filename = f"{unique_filename}_batch{b}.png"
                image_path = os.path.join(image_dir, image_filename)
                Image.fromarray(np_data_2d, mode='L').save(image_path)
                saved_files["image_paths"].append(image_path)
                print(f"Grayscale image saved to {image_path}")

            elif np_data.shape[0] == 3:  # RGB 图像
                # 转换 C,H,W 到 H,W,C
                np_data_rgb = np.transpose(np_data, (1, 2, 0))
                
                # 归一化和缩放到 0-255
                np_data_rgb = ((np_data_rgb - np_data_rgb.min()) / 
                               (np_data_rgb.max() - np_data_rgb.min()) * 255).astype(np.uint8)
                
                # 保存为PNG
                image_filename = f"{unique_filename}_batch{b}.png"
                image_path = os.path.join(image_dir, image_filename)
                Image.fromarray(np_data_rgb, mode='RGB').save(image_path)
                saved_files["image_paths"].append(image_path)
                print(f"RGB image saved to {image_path}")

            else:
                print(f"Unsupported channel count for batch {b}: {np_data.shape[0]}")

    # 增加全局计数器
    _SAVE_TENSOR_COUNTER += 1

    return saved_files