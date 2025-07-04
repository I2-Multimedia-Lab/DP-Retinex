import os
import time
import argparse
import warnings
import numpy as np
from glob import glob
from tqdm import tqdm
from pdb import set_trace as stx
from natsort import natsorted

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import cv2
from skimage import img_as_ubyte, metrics

import utils
from basicsr.models import create_model
from basicsr.utils.options import parse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="lpips")


def load_yaml(yaml_file):
    """Load YAML configuration file."""
    import yaml
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader
    
    with open(yaml_file, mode='r') as f:
        config = yaml.load(f, Loader=Loader)
    return config


def self_ensemble(x, model):
    """Apply self-ensemble technique to improve restoration quality."""
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        
        x = model(x)
        
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    
    transforms = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                transforms.append(forward_transformed(x, hflip, vflip, rot, model))
                
    transforms = torch.stack(transforms)
    return torch.mean(transforms, dim=0)


def process_standard_datasets(dataloader, model_restoration, args, factor, result_dir, result_dir_input, result_dir_gt):
    """Process SID, SMID, SDSD_indoor, or SDSD_outdoor datasets."""
    psnr = []
    ssim = []
    lpips = []
    total_time = 0
    total_frames = 0
    
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            target = data_batch['gt'].cpu().permute(0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Start timing
            start_time = time.time()
            
            # Padding in case images are not multiples of factor
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # End timing
            end_time = time.time()
            total_time += (end_time - start_time)
            total_frames += 1

            if args.GT_mean:
                # Use the mean of GT to rectify the output (same as KinD, LLFlow)
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            # Calculate metrics
            if args.compute_psnr:
                psnr.append(utils.PSNR(target, restored))
            if args.compute_ssim:
                ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))
            if args.compute_lpips:
                lpips.append(utils.calculate_lpips(target, restored))
                
            # Save results
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)

            output_basename = os.path.splitext(os.path.basename(inp_path))[0] + '.png'
            utils.save_img(os.path.join(result_dir, type_id, output_basename), img_as_ubyte(restored))
            utils.save_img(os.path.join(result_dir_input, type_id, output_basename), img_as_ubyte(input_save))
            utils.save_img(os.path.join(result_dir_gt, type_id, output_basename), img_as_ubyte(target))
    
    return psnr, ssim, lpips, total_time, total_frames


def process_other_datasets(input_dir, target_dir, model_restoration, args, factor, result_dir, output_dir):
    """Process other datasets with separate input and target images."""
    psnr = []
    ssim = []
    lpips = []
    total_time = 0
    total_frames = 0
    
    # Supported image formats
    image_formats = ['*.png', '*.jpg', '*.bmp', '*.JPG', '*.JPEG', '*.jpeg']
    
    # Get all input image paths
    input_paths = []
    for fmt in image_formats:
        input_paths.extend(glob(os.path.join(input_dir, fmt)))
    
    # Get all target image paths
    target_paths = []
    for fmt in image_formats:
        target_paths.extend(glob(os.path.join(target_dir, fmt)))
    
    # Remove any empty paths and sort
    input_paths = natsorted(list(filter(lambda x: x != '', input_paths)))
    target_paths = natsorted(list(filter(lambda x: x != '', target_paths)))
    
    # Check if paths are valid
    if not input_paths:
        raise ValueError(f"No input images found in {input_dir} for formats: {image_formats}")
    if not target_paths:
        raise ValueError(f"No target images found in {target_dir} for formats: {image_formats}")
    
    with torch.inference_mode():
        for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(target_paths)):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
    
            # Load images
            img = np.float32(utils.load_img_PIL(inp_path)) / 255.
            target = np.float32(utils.load_img_PIL(tar_path)) / 255.
    
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()
    
            # Resize if needed
            input_, original_size = utils.resize_image(input_, max_dim=1280)
    
            # Padding in case images are not multiples of factor
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    
            # Start timing
            start_time = time.time()
    
            # Process image (split for large images)
            if h < 1080 or w < 1080:
                if args.self_ensemble:
                    restored = self_ensemble(input_, model_restoration)
                else:
                    restored = model_restoration(input_)
            else:
                # Split and process
                input_1 = input_[:, :, :, 1::2]
                input_2 = input_[:, :, :, 0::2]
                if args.self_ensemble:
                    restored_1 = self_ensemble(input_1, model_restoration)
                    restored_2 = self_ensemble(input_2, model_restoration)
                else:
                    restored_1 = model_restoration(input_1)
                    restored_2 = model_restoration(input_2)
                restored = torch.zeros_like(input_)
                restored[:, :, :, 1::2] = restored_1
                restored[:, :, :, 0::2] = restored_2
    
            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
    
            # Resize back to original dimensions
            restored = F.interpolate(restored, size=original_size, mode='bicubic', align_corners=False)
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    
            # End timing
            end_time = time.time()
            total_time += (end_time - start_time)
            total_frames += 1
            
            if args.GT_mean:
                # Use the mean of GT to rectify the output
                mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
                restored = np.clip(restored * (mean_target / mean_restored), 0, 1)
    
            # Calculate metrics
            if args.compute_psnr:
                psnr.append(utils.calculate_psnr(target, restored))
            if args.compute_ssim:
                ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))
            if args.compute_lpips:
                lpips.append(utils.calculate_lpips(target, restored))
    
            # Save the restored image
            output_filename = os.path.basename(inp_path)
            if output_dir != '':
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = os.path.join(result_dir, output_filename)
            
            utils.save_img(output_path, img_as_ubyte(restored))
    
    return psnr, ssim, lpips, total_time, total_frames


def main():
    """Main function that handles the entire testing process."""
    parser = argparse.ArgumentParser(description='Image Enhancement using Retinexformer')
    
    # Input/output arguments
    parser.add_argument('--input_dir', default='./Enhancement/Datasets', type=str, 
                        help='Directory of validation images')
    parser.add_argument('--result_dir', default='./results/', type=str, 
                        help='Directory for results')
    parser.add_argument('--output_dir', default='', type=str, 
                        help='Directory for output')
    
    # Model configuration arguments
    parser.add_argument('--opt', type=str, default='Options/No_inference_img_enhancement_.yml', 
                        help='Path to option YAML file')
    parser.add_argument('--weights', default='pretrained_weights/best_psnr_23.49_47000.pth', 
                        type=str, help='Path to weights')
    parser.add_argument('--dataset', default='Camera_Captured', type=str, 
                        help='Test Dataset')
    
    # Processing arguments
    parser.add_argument('--gpus', type=str, default="0", 
                        help='GPU devices')
    parser.add_argument('--GT_mean', action='store_true', 
                        help='Use the mean of GT to rectify the output of the model')
    parser.add_argument('--self_ensemble', action='store_true', 
                        help='Use self-ensemble to obtain better results')
    
    # Evaluation metrics arguments
    parser.add_argument('--compute_psnr', action='store_true', default=True, 
                        help='whether to compute PSNR')
    parser.add_argument('--compute_ssim', action='store_true', default=True, 
                        help='whether to compute SSIM')
    parser.add_argument('--compute_lpips', action='store_true', default=True, 
                        help='whether to compute LPIPS')
    
    args = parser.parse_args()
    
    # Set CUDA visible devices
    gpu_list = ','.join(str(x) for x in args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    
    # Load YAML configuration
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    
    # Load model type from configuration
    yaml_config = load_yaml(args.opt)
    model_type = yaml_config['network_g'].pop('type')
    
    # Create and load model
    model_restoration = create_model(opt).net_g
    checkpoint = torch.load(args.weights)
    
    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)
    
    print(f"===>Testing using weights: {args.weights}")
    
    # Prepare model for testing
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    
    # Set up directory structure
    factor = 8
    dataset = args.dataset
    config = os.path.basename(args.opt).split('.')[0]
    checkpoint_name = os.path.basename(args.weights).split('.')[0]
    
    result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
    result_dir_input = os.path.join(args.result_dir, dataset, 'input')
    result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
    output_dir = args.output_dir
    
    os.makedirs(result_dir, exist_ok=True)
    if args.output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    
    # Process dataset according to type
    if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
        os.makedirs(result_dir_input, exist_ok=True)
        os.makedirs(result_dir_gt, exist_ok=True)
        
        # Import appropriate dataset class
        if dataset == 'SID':
            from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
        elif dataset == 'SMID':
            from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
        else:
            from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
        
        # Configure dataset
        data_opt = opt['datasets']['val']
        data_opt['phase'] = 'test'
        if data_opt.get('scale') is None:
            data_opt['scale'] = 1
            
        # Fix paths with expanduser if needed
        if '~' in data_opt['dataroot_gt']:
            data_opt['dataroot_gt'] = os.path.expanduser('~') + data_opt['dataroot_gt'][1:]
        if '~' in data_opt['dataroot_lq']:
            data_opt['dataroot_lq'] = os.path.expanduser('~') + data_opt['dataroot_lq'][1:]
        
        # Create dataset and dataloader
        test_dataset = Dataset(data_opt)
        print(f'Test dataset length: {len(test_dataset)}')
        dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        
        # Process images
        psnr_list, ssim_list, lpips_list, total_time, total_frames = process_standard_datasets(
            dataloader, model_restoration, args, factor, result_dir, result_dir_input, result_dir_gt
        )
    else:
        # Process other datasets
        input_dir = opt['datasets']['val']['dataroot_lq']
        target_dir = opt['datasets']['val']['dataroot_gt']
        print(f"Input directory: {input_dir}")
        print(f"Target directory: {target_dir}")
        
        psnr_list, ssim_list, lpips_list, total_time, total_frames = process_other_datasets(
            input_dir, target_dir, model_restoration, args, factor, result_dir, output_dir
        )
    
    # Calculate and print metrics
    def print_metric(metric_name, metric_list, compute_flag):
        if compute_flag and metric_list:
            metric_value = np.mean(np.array(metric_list))
            print(f"{metric_name}: {metric_value:.6f}")
        else:
            print(f"{metric_name} metric is not computed")
    
    print_metric("PSNR", psnr_list, args.compute_psnr)
    print_metric("SSIM", ssim_list, args.compute_ssim)
    print_metric("LPIPS", lpips_list, args.compute_lpips)
    
    # Calculate and print average processing time and FPS
    if total_frames > 0:
        avg_time_per_image = total_time / total_frames
        fps = total_frames / total_time if total_time > 0 else 0
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total frames processed: {total_frames}")
        print(f"Average time per image: {avg_time_per_image:.4f} seconds")
        print(f"Average FPS: {fps:.2f}")
    else:
        print("No frames were processed.")


if __name__ == "__main__":
    main()