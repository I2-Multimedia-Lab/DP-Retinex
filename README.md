# 🌙 DP-Retinex: Dual-Prior Guided Low-Light Image Enhancement with YUV-Domain Reflectance-Illumination Decomposition

Welcome to **DP-Retinex**! This repository provides a powerful, Retinex-theory-inspired image enhancement model built with PyTorch 🧠. Whether you're working on night-time photography, autonomous vision, or general low-level vision tasks, this toolkit is made for you! 💡

---

## 🚀 Installation Guide

### ✅ Step 1: Create a Conda Environment

```bash
conda create -n DP-Retinex python=3.8
conda activate DP-Retinex
````

> 🧊 Python 3.8 is recommended for compatibility.

---

### 🔧 Step 2: Install Core Dependencies

```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

> 🚀 This installs PyTorch with CUDA 11.7 support (for GPU acceleration).

---

### 📦 Step 3: Install Additional Requirements

```bash
pip install -r requirements.txt
```

This includes packages like `numpy`, `opencv-python`, `scikit-image`, `tqdm`, etc.

---

### 🛠️ Step 4: Develop Mode Installation

```bash
python setup.py develop --no_cuda_ext
```

> 🔧 Use `--no_cuda_ext` to skip compiling CUDA ops (optional, for environments without NVCC or GPU).

---

## 📂 Project Structure (简要)

```
DP-Retinex/
├── basicsr/                      # Core image restoration framework (based on BasicSR)
├── Enhancement/                  # Scripts for evaluation and inference (test_from_dataset.py, etc.)
├── Options/                      # YAML configuration files for different datasets & experiments
├── pretrained_weights/           # (You should manually place downloaded model weights here)

├── data/                         # (You should place LOLv1/LOLv2/SDSD datasets here)
  
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup file for develop mode
└── README.md                     # You're here 😄

```

## 📥 Dataset & Weights Download

### 🗂️ LOLv1 & LOLv2 Dataset

To evaluate our model, please download the **LOLv1** and **LOLv2** datasets from the following Google Drive link:

🔗 [Download LOLv1 / LOLv2 Dataset](https://drive.google.com/file/d/1cKj4t45AxGJjK4Z2v-T1hMjhSnSNlJjC/view?usp=drive_link) (📁 Google Drive)

Please place the datasets under:

```bash
data/LOLv1/
data/LOLv2/
```

---

### 💾 Pretrained Weights

We provide several pretrained models trained on different benchmarks. You can download all weights from the following Google Drive folder:

🔗 [Download Pretrained Weights](https://drive.google.com/drive/folders/1-kyiMzUDK6PN9Nafo7Jni3v5xjwlCUFf?usp=drive_link) (📁 Google Drive)

Please organize them as:

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

## ⚡ Quick Testing

Once your environment is set up and models are downloaded, you can quickly run inference using the following commands:

### ▶️ LOLv1

```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_LOL_v1.yml \
  --weights pretrained_weights/LOLv1_best_psnr_23.48_109000.pth
```

---

### ▶️ LOLv2

**Real subset**:

```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_LOL_v2_real.yml \
  --weights pretrained_weights/LOLv2_real_best_psnr_23.49_47000.pth \
  --dataset LOL_v2_real
```

**Synthetic subset**:

```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_LOL_v2_syn.yml \
  --weights pretrained_weights/LOLv2_syn_bet_psnr_26.30_229000.pth \
  --dataset LOL_v2_syn
```

---

### ▶️ SDSD Benchmark

**Indoor**:

```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_SDSD_indoorv2.yml \
  --weights pretrained_weights/SDSD_indoor_best_psnr_30.70_45000.pth \
  --dataset SDSD_indoor
```

**Outdoor**:

```bash
python3 Enhancement/test_from_dataset.py \
  --opt Options/Retinex_Degradation_SDSD_outdoorv2.yml \
  --weights pretrained_weights/SDSD_outdoor_best_psnr_29.57_60000.pth \
  --dataset SDSD_outdoor
```

---

## 🧪 Inference Results

All outputs will be saved under:

```
results/
└── [DatasetName]/
     └── Enhanced/
          ├── image_0001.png
          ├── ...
```

Each configuration file (`*.yml`) controls test resolution, padding mode, input/output paths, and model hyperparameters.

---


## 💬 Tips & Troubleshooting

* ❓ **Missing NVCC?** Try `--no_cuda_ext` as above.
* 🧪 Test if PyTorch is installed with GPU:

```python
import torch
print(torch.cuda.is_available())
```
