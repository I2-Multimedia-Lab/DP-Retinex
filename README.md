# 🌙 DP-Retinex: Dual-Prior Guided Low-Light Image Enhancement with YUV-Domain Reflectance-Illumination Decomposition
# 🌙 DP-Retinex：基于YUV域反射-光照分解的双先验引导低光图像增强

Welcome to **DP-Retinex**! This repository provides a powerful, Retinex-theory-inspired image enhancement model built with PyTorch 🧠. Whether you're working on night-time photography, autonomous vision, or general low-level vision tasks, this toolkit is made for you! 💡

欢迎使用 **DP-Retinex**！本仓库提供了一个基于 Retinex 理论的强大图像增强模型，使用 PyTorch 构建 🧠。无论你是在做夜间摄影、自动驾驶视觉还是通用底层视觉任务，这个工具包都适合你！💡

---

## 📥 Dataset & Weights Download | 数据集与权重下载

### 🗂️ LOLv1 & LOLv2 Dataset | LOLv1 和 LOLv2 数据集

To evaluate our model, please download the **LOLv1** and **LOLv2** datasets from the following links:

为了评估我们的模型，请从以下链接下载 **LOLv1** 和 **LOLv2** 数据集。

#### LOLv1
🔗 **Google Drive**:  https://drive.google.com/file/d/16ShfICyRRf2bDaTW8JnP87Sq4abha84U/view?usp=sharing

#### LOLv2
🔗 **Google Drive**:  https://drive.google.com/file/d/1cKj4t45AxGJjK4Z2v-T1hMjhSnSNlJjC/view?usp=drive_link

🔗 **Baidu Netdisk | 百度网盘**: [https://pan.baidu.com/s/11VeVUmJKovOJsYqS3v2Tkw](https://pan.baidu.com/s/11VeVUmJKovOJsYqS3v2Tkw)  
**提取码 | Extraction Code**: `9vph`

Please place the datasets under: | 请将数据集放置在以下目录：
```bash
data/LOLv1/
data/LOLv2/
```

---

### 💾 Pretrained Weights | 预训练权重

We provide several pretrained models trained on different benchmarks. You can download all weights from the following links:

我们提供了在不同基准数据集上训练的多个预训练模型。你可以从以下链接下载所有权重：

🔗 **Google Drive | 谷歌云盘**: [Download Pretrained Weights](https://drive.google.com/drive/folders/1-kyiMzUDK6PN9Nafo7Jni3v5xjwlCUFf?usp=drive_link)

🔗 **Baidu Netdisk | 百度网盘**: [https://pan.baidu.com/s/11VeVUmJKovOJsYqS3v2Tkw](https://pan.baidu.com/s/11VeVUmJKovOJsYqS3v2Tkw)  
**提取码 | Extraction Code**: `9vph`

Please organize them as: | 请按以下方式组织权重文件：
```bash
pretrained_weights/
│
├── LOLv1_best_psnr_23.48_109000.pth
├── LOLv2_real_best_psnr_23.49_47000.pth
├── LOLv2_syn_bet_psnr_26.30_229000.pth
├── SDSD_indoor_best_psnr_30.70_45000.pth
└── SDSD_outdoor_best_psnr_29.57_60000.pth
```

---

## 🚀 Installation Guide | 安装指南

### ✅ Step 1: Create a Conda Environment | 步骤1：创建 Conda 环境
```bash
conda create -n DP-Retinex python=3.8
conda activate DP-Retinex
```

> 🧊 Python 3.8 is recommended for compatibility. | 推荐使用 Python 3.8 以确保兼容性。

---

### 🔧 Step 2: Install Core Dependencies | 步骤2：安装核心依赖
```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

> 🚀 This installs PyTorch with CUDA 11.7 support (for GPU acceleration). | 这将安装支持 CUDA 11.7 的 PyTorch（用于 GPU 加速）。

---

### 📦 Step 3: Install Additional Requirements | 步骤3：安装额外依赖
```bash
pip install -r requirements.txt
```

This includes packages like `numpy`, `opencv-python`, `scikit-image`, `tqdm`, etc.

这包括 `numpy`、`opencv-python`、`scikit-image`、`tqdm` 等软件包。

---

### 🛠️ Step 4: Develop Mode Installation | 步骤4：开发模式安装
```bash
python setup.py develop --no_cuda_ext
```

> 🔧 Use `--no_cuda_ext` to skip compiling CUDA ops (optional, for environments without NVCC or GPU). | 使用 `--no_cuda_ext` 跳过 CUDA 算子编译（可选，适用于没有 NVCC 或 GPU 的环境）。

---

## 📂 Project Structure | 项目结构
```
DP-Retinex/
├── basicsr/                      # Core image restoration framework (基于 BasicSR 的核心图像恢复框架)
├── Enhancement/                  # Evaluation and inference scripts (评估和推理脚本)
├── Options/                      # YAML configuration files (YAML 配置文件)
├── pretrained_weights/           # Place downloaded model weights here (下载的模型权重放置于此)
├── data/                         # Place datasets here (数据集放置于此)
├── requirements.txt              # Python dependencies (Python 依赖)
├── setup.py                      # Package setup file (包安装文件)
└── README.md                     # You're here 😄 (你在这里 😄)
```

---

## ⚡ Quick Testing | 快速测试

Once your environment is set up and models are downloaded, you can quickly run inference using the following commands:

环境设置完成并下载模型后，你可以使用以下命令快速运行推理：

### ▶️ LOLv1
```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_LOL_v1.yml \
  --weights pretrained_weights/LOLv1_best_psnr_23.48_109000.pth
```

---

### ▶️ LOLv2

**Real subset | 真实子集**:
```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_LOL_v2_real.yml \
  --weights pretrained_weights/LOLv2_real_best_psnr_23.49_47000.pth \
  --dataset LOL_v2_real
```

**Synthetic subset | 合成子集**:
```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_LOL_v2_syn.yml \
  --weights pretrained_weights/LOLv2_syn_bet_psnr_26.30_229000.pth \
  --dataset LOL_v2_syn
```

---

### ▶️ SDSD Benchmark | SDSD 基准测试

**Indoor | 室内**:
```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_SDSD_indoorv2.yml \
  --weights pretrained_weights/SDSD_indoor_best_psnr_30.70_45000.pth \
  --dataset SDSD_indoor
```

**Outdoor | 室外**:
```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_SDSD_outdoorv2.yml \
  --weights pretrained_weights/SDSD_outdoor_best_psnr_29.57_60000.pth \
  --dataset SDSD_outdoor
```

---

## 🧪 Inference Results | 推理结果

All outputs will be saved under: | 所有输出将保存在：
```
results/
└── [DatasetName]/
     └── Enhanced/
          ├── image_0001.png
          ├── ...
```

Each configuration file (`*.yml`) controls test resolution, padding mode, input/output paths, and model hyperparameters.

每个配置文件（`*.yml`）控制测试分辨率、填充模式、输入/输出路径和模型超参数。

---

## 💬 Tips & Troubleshooting | 提示与故障排除

* ❓ **Missing NVCC? | 缺少 NVCC？** Try `--no_cuda_ext` as above. | 尝试使用上面的 `--no_cuda_ext`。
* 🧪 **Test if PyTorch is installed with GPU | 测试 PyTorch 是否支持 GPU**:
```python
import torch
print(torch.cuda.is_available())
```

---

## 📧 Contact | 联系方式

For questions or issues, please open an issue on GitHub or contact us via email.

如有问题，请在 GitHub 上提交 issue 或通过电子邮件联系我们。

---

## 📄 Citation | 引用

If you find this work useful, please consider citing our paper:

如果你觉得这项工作有用，请考虑引用我们的论文：
```bibtex
@ARTICLE{11339941,
  author={Zhao, Zhengkai and Gao, Longmi and Gao, Pan and Qin, Jie},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={DP-Retinex: Dual-Prior Guided Low-Light Image Enhancement with YUV-Domain Reflectance-Illumination Decomposition}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Image color analysis;Lighting;Image enhancement;Feature extraction;Transformers;Image restoration;Reflectivity;Nonlinear distortion;Diffusion models;Degradation;Low-Light Image Enhancement;Dual Priors;YUV Color Space;Diffusion-Transformer Hybrid},
  doi={10.1109/TCSVT.2026.3651859}}

```

---

## 📜 License | 许可证

This project is licensed under the MIT License.

本项目采用 MIT 许可证。
