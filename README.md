## 📘 README.md（建议内容如下）

# 🌙 DP-Retinex: A Diffusion-Based Retinex Framework for Low-Light Image Enhancement

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
├── basicsr/               # Core image restoration framework
├── data/                  # Dataset definitions
├── options/               # Training/validation config files
├── scripts/               # Utility and batch scripts
├── setup.py               # For develop mode installation
├── requirements.txt       # Dependency list
└── README.md              # You're here 😉
```

---

## 💬 Tips & Troubleshooting

* ❓ **Missing NVCC?** Try `--no_cuda_ext` as above.
* 🧪 Test if PyTorch is installed with GPU:

```python
import torch
print(torch.cuda.is_available())
```
