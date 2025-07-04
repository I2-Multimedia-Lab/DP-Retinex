import os
import argparse
import glob
import numpy as np
import torch
import pyiqa
from tqdm import tqdm
from PIL import Image
import warnings
import numpy as np

# 抑制不必要的警告
warnings.filterwarnings("ignore")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算图像的 NIQE 和 BRISQUE 质量指标')
    parser.add_argument('--folders', nargs='+', required=True, help='要评估的文件夹路径，可以包含通配符')
    parser.add_argument('--output_csv', default='image_quality_metrics.csv', help='CSV 输出文件路径')
    parser.add_argument('--verbose', action='store_true', help='是否显示每张图片的详细指标')
    return parser.parse_args()

def calculate_niqe(image_path):
    """
    计算 NIQE 指标，使用 pyiqa 库。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        float: NIQE 分数。
    """
    try:
        # 创建 NIQE 评估器
        niqe_metric = pyiqa.create_metric('niqe')

        # 使用 PIL 加载图像
        img = Image.open(image_path).convert('RGB')

        # 转换为 numpy 数组
        img_array = np.array(img)

        # 转换为 PyTorch 张量，并调整维度顺序
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # 如果 CUDA 可用，则将张量移动到 GPU
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            niqe_metric = niqe_metric.cuda()

        # 计算 NIQE 分数
        niqe_value = niqe_metric(img_tensor).item()

        return niqe_value
    except Exception as e:
        print(f"计算 NIQE 时出错 ({image_path}): {e}")
        return None

def calculate_brisque(image_path):
    """
    计算 BRISQUE 指标，使用 pyiqa 库。

    Args:
        image_path (str): 图像文件路径。

    Returns:
        float: BRISQUE 分数。
    """
    try:
        # 创建 BRISQUE 评估器
        brisque_metric = pyiqa.create_metric('brisque')

        # 使用 PIL 加载图像
        img = Image.open(image_path).convert('RGB')

        # 转换为 numpy 数组
        img_array = np.array(img)

        # 转换为 PyTorch 张量，并调整维度顺序
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # 如果 CUDA 可用，则将张量移动到 GPU
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            brisque_metric = brisque_metric.cuda()

        # 计算 BRISQUE 分数
        brisque_value = brisque_metric(img_tensor).item()

        return brisque_value
    except Exception as e:
        print(f"计算 BRISQUE 时出错 ({image_path}): {e}")
        return None

def calculate_metrics_for_folder(folder_pattern):
    """计算文件夹中所有图像的质量指标"""
    # 展开通配符
    image_paths = []
    
    # 检查是否是目录或通配符模式
    if os.path.isdir(folder_pattern):
        # 如果是目录，获取所有支持的图像格式
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
            image_paths.extend(glob.glob(os.path.join(folder_pattern, ext)))
    else:
        # 否则假定是通配符模式
        image_paths = glob.glob(folder_pattern)
    
    if not image_paths:
        print(f"警告: 没有找到符合模式 '{folder_pattern}' 的图像")
        return None, None, []
    
    niqe_scores = []
    brisque_scores = []
    image_results = []
    
    for image_path in tqdm(image_paths, desc=f"处理 {folder_pattern}"):
        # 确保是文件而不是目录
        if os.path.isfile(image_path):
            # 计算指标
            niqe = calculate_niqe(image_path)
            brisque = calculate_brisque(image_path)
            
            # 记录有效的分数
            if niqe is not None:
                niqe_scores.append(niqe)
            if brisque is not None:
                brisque_scores.append(brisque)
                
            # 保存每张图片的结果
            image_name = os.path.basename(image_path)
            image_results.append({
                'image': image_name,
                'niqe': niqe,
                'brisque': brisque
            })
    
    # 计算平均分数
    avg_niqe = np.mean(niqe_scores) if niqe_scores else None
    avg_brisque = np.mean(brisque_scores) if brisque_scores else None
    
    return avg_niqe, avg_brisque, image_results

def main():
    """主函数"""
    args = parse_args()
    results = []
    all_image_results = []
    
    # 遍历用户传入的多个文件夹路径
    for folder in args.folders:
        print(f"\n评估文件夹: {folder}")
        avg_niqe, avg_brisque, image_results = calculate_metrics_for_folder(folder)
        
        if avg_niqe is not None and avg_brisque is not None:
            print(f"文件夹: {folder} | 平均 NIQE: {avg_niqe:.4f}, 平均 BRISQUE: {avg_brisque:.4f}")
            results.append((folder, avg_niqe, avg_brisque, len(image_results)))
            
            # 如果用户指定了详细模式，显示每张图片的指标
            if args.verbose:
                print("\n详细指标:")
                for res in image_results:
                    if res['niqe'] is not None and res['brisque'] is not None:
                        print(f"  {res['image']}: NIQE = {res['niqe']:.4f}, BRISQUE = {res['brisque']:.4f}")
            
            # 将每张图片的结果添加到总结果
            for res in image_results:
                if res['niqe'] is not None and res['brisque'] is not None:
                    all_image_results.append({
                        'folder': folder,
                        'image': res['image'],
                        'niqe': res['niqe'],
                        'brisque': res['brisque']
                    })
        else:
            print(f"跳过文件夹 {folder} (没有有效图像)")

    # 总结所有结果
    print("\n评估摘要:")
    for folder, niqe, brisque, count in results:
        print(f"{folder} ({count} 张图片): NIQE = {niqe:.4f}, BRISQUE = {brisque:.4f}")

    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

# outputs/DICM/LOLv1_best_psnr_23/*.png: NIQE = 3.8355, BRISQUE = 17.6293
# outputs/DICM/LOLv2_real_best_psnr_23/*.png: NIQE = 4.2751, BRISQUE = 17.9753
# outputs/DICM/LOLv2_syn_bet_psnr_26/*.png: NIQE = 3.8866, BRISQUE = 18.3585

# outputs/LIME/LOLv1_best_psnr_23/*.png: NIQE = 4.5761, BRISQUE = 26.5361
# outputs/LIME/LOLv2_real_best_psnr_23/*.png: NIQE = 4.5594, BRISQUE = 16.0143
# outputs/LIME/LOLv2_syn_bet_psnr_26/*.png: NIQE = 4.2125, BRISQUE = 18.3448

# outputs/MEF/LOLv1_best_psnr_23/*.png: NIQE = 3.6339, BRISQUE = 12.7021
# outputs/MEF/LOLv2_real_best_psnr_23/*.png: NIQE = 3.8132, BRISQUE = 15.1851
# outputs/MEF/LOLv2_syn_bet_psnr_26/*.png: NIQE = 3.6604, BRISQUE = 14.2761

# outputs/NPE/LOLv1_best_psnr_23/*.png: NIQE = 3.8094, BRISQUE = 19.4674
# outputs/NPE/LOLv2_real_best_psnr_23/*.png: NIQE = 3.8591, BRISQUE = 17.2420
# outputs/NPE/LOLv2_syn_bet_psnr_26/*.png: NIQE = 3.6826, BRISQUE = 21.2937

# outputs/VV/LOLv1_best_psnr_23/*.png: NIQE = 4.6320, BRISQUE = 36.9775
# outputs/VV/LOLv2_real_best_psnr_23/*.png: NIQE = 4.8283, BRISQUE = 36.8709
# outputs/VV/LOLv2_syn_bet_psnr_26/*.png: NIQE = 4.9934, BRISQUE = 38.9052
# outputs/VV/SDSD_indoor_best_psnr_30/*.png: NIQE = 5.0752, BRISQUE = 34.5665
# outputs/VV/SDSD_outdoor_best_psnr_29/*.png: NIQE = 4.9500, BRISQUE = 30.5687


'''
# 评估单个文件夹
python mEnhancement/measure_niqe_bris.py --folders "outputs/DICM/*.png"

# 评估多个文件夹
python Enhancement/measure_niqe_bris.py --folders "outputs/DICM/*.png" "outputs/LIME/*.png" "outputs/MEF/*.png"

# 显示每张图片的详细指标
python Enhancement/measure_niqe_bris.py --folders "results/NPE/No_inference_img_enhancement_/LOLv1_best_psnr_23/*.png" --verbose

# 自定义输出 CSV 文件名
python Enhancement/measure_niqe_bris.py --folders "outputs/DICM/*.png" --output_csv results.csv
'''