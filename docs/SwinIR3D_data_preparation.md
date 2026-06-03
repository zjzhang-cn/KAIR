# SwinIR3D 训练数据准备指南

## 概述

SwinIR3D 用于 3D 体数据（CT、MRI、显微镜等）的超分辨率、去噪等任务。与 2D 图像训练不同，3D 训练需要准备体数据（volumetric data），本文档详细说明数据格式、预处理步骤、目录组织和训练配置。

---

## 1. 支持的数据格式

| 格式 | 后缀 | 说明 | 推荐度 |
|------|------|------|:---:|
| NumPy | `.npy` | 单个体数据，shape `(D, H, W)` 或 `(C, D, H, W)` | ★★★ |
| NumPy 压缩 | `.npz` | 可包含多个数组，取第一个 | ★★☆ |
| NIfTI | `.nii` / `.nii.gz` | 医学影像标准格式，需安装 `nibabel` | ★★★ |

所有格式在加载时自动归一化到 `[0, 1]`。

---

## 2. 目录结构

```
project/
├── trainsets/
│   ├── trainH_3d/              # HR 体数据目录（必须）
│   │   ├── vol_001.npy         # shape: (32, 256, 256)  各向同性 3D
│   │   ├── vol_002.nii.gz      # 支持混合格式
│   │   └── ...
│   └── trainL_3d/              # LR 体数据目录（可选）
│       └── ...                 # 不提供则自动从 HR 下采样生成
│
├── testsets/
│   └── testH_3d/               # 测试集
│       ├── test_001.npy
│       └── ...
│
└── options/
    └── train_swinir3d_sr.json  # 训练配置
```

**两种数据组织方式：**

| 方式 | dataroot_H | dataroot_L | 说明 |
|------|-----------|-----------|------|
| **A（推荐）** | 提供 HR | `null`（不提供） | 训练时自动对 HR 下采样生成 LR，省去手动准备配对数据 |
| **B** | 提供 HR | 提供 LR | 已有精确配对的 L/H 数据，文件需一一对应且命名一致 |

---

## 3. 公开数据集推荐

### 医学影像

| 数据集 | 模态 | 典型尺寸 | 适用任务 | 链接 |
|--------|------|---------|---------|------|
| **HCP** | 脑 MRI T1/T2 | 256³, 1mm iso | 3D 各向同性 SR | [humanconnectome.org](https://www.humanconnectome.org/) |
| **ADNI** | 脑 MRI/PET | 192×192×166 | 3D SR、去噪 | [adni.loni.usc.edu](https://adni.loni.usc.edu/) |
| **LIDC-IDRI** | 胸部 CT | 512×512×N, 层厚 1-5mm | 厚层→薄层 (aniso_xy) | [wiki.cancerimagingarchive.net](https://wiki.cancerimagingarchive.net/) |
| **LUNA16** | 胸部 CT | 512×512×N | 各向异性 SR | [luna16.grand-challenge.org](https://luna16.grand-challenge.org/) |
| **KiTS** | 腹部 CT | 512×512×N | 3D SR | [kits-challenge.org](https://kits-challenge.org/) |
| **CHAOS** | 腹部 MRI T1/T2 | 各向异性 | Z 方向 SR (aniso_z) | [chaos.grand-challenge.org](https://chaos.grand-challenge.org/) |
| **BraTS** | 脑 MRI 多模态 | 240×240×155 | 3D SR、去噪 | [braintumorsegmentation.org](http://braintumorsegmentation.org/) |

### 显微镜 / 其他

| 数据集 | 模态 | 典型尺寸 | 说明 |
|--------|------|---------|------|
| **Cell Tracking Challenge** | 荧光显微镜 | 各向异性 | Z 分辨率通常远低于 XY |
| **CREMI** | 电子显微镜 | 125×1250×1250 | 神经元连接组学 |
| **OASIS** | 脑 MRI | 256³ | 各向同性，适合 SR 基准 |

---

## 4. 数据预处理

### 4.1 加载原始数据

```python
import numpy as np
import nibabel as nib  # 用于 .nii / .nii.gz

# NIfTI 格式
nii = nib.load("scan.nii.gz")
vol = nii.get_fdata()  # shape: (H, W, D) 注意维度顺序！

# 转为标准 DHW 顺序
vol = np.transpose(vol, (2, 0, 1))  # (H, W, D) → (D, H, W)
```

### 4.2 归一化到 [0, 1]

```python
vol = vol.astype(np.float32)

# Min-Max 归一化
vol_min, vol_max = vol.min(), vol.max()
vol = (vol - vol_min) / (vol_max - vol_min)

# 或：CT 窗宽窗位归一化（HU 值 → [0,1]）
# hu_min, hu_max = -1000, 400  # 肺窗
# vol = np.clip((vol - hu_min) / (hu_max - hu_min), 0, 1)
```

### 4.3 各向同性重采样（可选但推荐）

当原始数据各向异性时（如 CT 层间距 5mm, 面内 0.5mm），建议先重采样为各向同性以便统一训练：

```python
from scipy.ndimage import zoom

# 原始体素尺寸 (mm): (spacing_Z, spacing_Y, spacing_X)
orig_spacing = (5.0, 0.5, 0.5)   # 典型厚层 CT
target_spacing = (1.0, 1.0, 1.0)  # 目标各向同性

# 计算缩放因子
D, H, W = vol.shape
zoom_factors = (
    orig_spacing[0] / target_spacing[0],  # Z: 5.0/1.0 = 5.0x
    orig_spacing[1] / target_spacing[1],  # Y: 0.5/1.0 = 0.5x
    orig_spacing[2] / target_spacing[2],  # X: 0.5/1.0 = 0.5x
)

vol_iso = zoom(vol, zoom_factors, order=3)  # cubic 插值
# 新尺寸: (D*5, H*0.5, W*0.5)
```

> **注意**：如果目标恰好是各向异性 SR（如厚层→薄层），则不需要重采样，保留原生各向异性分辨率，在训练时用 `sr3d_aniso_xy` 或 `sr3d_aniso_z` 模式。

### 4.4 切片提取（大数据集）

对于过大的体数据（如 512×512×2000），应提取小块：

```python
def extract_subvolumes(vol, sub_size=(64, 128, 128), stride=(32, 96, 96)):
    """从大体数据中提取子块，增加数据量"""
    D, H, W = vol.shape
    sD, sH, sW = sub_size
    stD, stH, stW = stride
    patches = []
    for d in range(0, D - sD + 1, stD):
        for h in range(0, H - sH + 1, stH):
            for w in range(0, W - sW + 1, stW):
                patch = vol[d:d+sD, h:h+sH, w:w+sW]
                # 过滤：跳过信息量低的区域（如全黑背景）
                if patch.std() > 0.01:
                    patches.append(patch)
    return patches

# 每个子块单独存为一个 .npy 作为训练样本
for i, patch in enumerate(extract_subvolumes(vol)):
    np.save(f"trainsets/trainH_3d/vol_{i:04d}.npy", patch)
```

### 4.5 完整的预处理脚本模板

```python
import os, sys
import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, '..')
from data.dataset_3d import load_volume, downsample_3d

def preprocess_3d_dataset(input_dir, output_dir, target_spacing=None):
    """
    批量预处理 3D 体数据
    Args:
        input_dir: 原始数据目录（.nii.gz 或 .npy）
        output_dir: 输出目录
        target_spacing: 目标体素尺寸 (d, h, w) mm，None 则保持原尺寸
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(('.nii', '.nii.gz', '.npy')):
            continue

        print(f"处理: {fname}")
        vol = load_volume(os.path.join(input_dir, fname))

        # 归一化
        vol = vol.astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min())

        # 重采样（可选）
        if target_spacing is not None:
            orig_shape = np.array(vol.shape[-3:])
            target_shape = orig_shape * target_spacing  # 根据实际体素尺寸调整
            zoom_factors = target_shape / orig_shape
            vol = np.stack([zoom(vol[i], zoom_factors, order=3)
                           for i in range(vol.shape[0])])

        # 确保 float32 并保存
        vol = vol.astype(np.float32)
        output_name = os.path.splitext(fname)[0].replace('.nii', '')
        np.save(os.path.join(output_dir, f"{output_name}.npy"), vol)

    print(f"完成！共处理文件，保存至 {output_dir}")

if __name__ == '__main__':
    preprocess_3d_dataset(
        input_dir='raw_data/CT_scans/',
        output_dir='trainsets/trainH_3d/'
    )
```

---

## 5. 训练配置说明

### 5.1 完整配置模板

配置文件：`options/train_swinir3d_sr.json`

```json
{
  "task": "swinir3d_sr",
  "model": "plain",
  "gpu_ids": [0],

  "scale": [2, 2, 2],          // (sD, sH, sW) — 支持各向异性
  "n_channels": 1,              // CT/MRI=1, RGB视频=3

  "datasets": {
    "train": {
      "dataset_type": "sr3d",   // 下采样模式（见 5.3）
      "dataroot_H": "trainsets/trainH_3d",
      "dataroot_L": null,       // null=自动从HR下采样生成LR
      "H_size": [16, 64, 64],   // HR patch 大小 (D, H, W)
      "dataloader_batch_size": 1
    },
    "test": {
      "dataset_type": "sr3d",
      "dataroot_H": "testsets/testH_3d",
      "dataroot_L": null
    }
  },

  "netG": {
    "net_type": "swinir3d",
    "img_size": [16, 64, 64],
    "in_chans": 1,
    "embed_dim": 48,            // 48=小模型, 96=大模型
    "depths": [4, 4, 4, 4],
    "num_heads": [4, 4, 4, 4],
    "window_size": [2, 4, 4],   // 3D窗口: D方向小(Z分辨率低)
    "mlp_ratio": 2,
    "upscale": [2, 2, 2],
    "upsampler": "pixelshuffle3d",
    "img_range": 1.0
  },

  "train": {
    "G_lossfn_type": "l1",
    "G_lossfn_weight": 1.0,
    "G_optimizer_lr": 1e-4,
    "G_scheduler_milestones": [50000, 100000, 200000, 300000],
    "G_scheduler_gamma": 0.5,
    "checkpoint_test": 5000,
    "checkpoint_save": 5000,
    "checkpoint_print": 100
  }
}
```

### 5.2 关键参数速查

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `H_size[0]` (D) | 8-32 | 深度方向 patch 大小，受 GPU 显存限制 |
| `H_size[1,2]` (H,W) | 32-128 | 面内 patch 大小 |
| `window_size[0]` (wD) | 2-4 | D 方向窗口，一般 ≤ 物理层数分辨率 |
| `window_size[1,2]` (wH,wW) | 4-8 | 面内窗口，同 2D SwinIR |
| `embed_dim` | 48 (小) / 96 (大) | 越大效果越好，显存也越大 |
| `dataloader_batch_size` | 1-2 | 3D 显存消耗大，从 1 起步 |
| `G_optimizer_lr` | 1e-4 | Adam 标准学习率 |

### 5.3 dataset_type 与自动下采样

| dataset_type | H_size | scale | 自动下采样行为 | 典型场景 |
|-------------|--------|-------|--------------|---------|
| `sr3d` | `[16,64,64]` | `[2,2,2]` | D,H,W 同时 ↓2x | 各向同性 3D SR |
| `sr3d` | `[32,64,64]` | `[4,2,2]` | D↓4x, H,W↓2x | 非对称上采样 |
| `sr3d_aniso_xy` | `[16,64,64]` | `[1,2,2]` | Z不变, XY↓2x | 厚层CT→薄层 |
| `sr3d_aniso_z` | `[64,64,64]` | `[4,1,1]` | XY不变, Z↓4x | Z方向增强 |
| `denoise3d` | `[16,64,64]` | `1` | 加高斯噪声 | 3D 去噪 |
| `plain3d` | `[16,64,64]` | `1` | 不生成LR（需提供L路径） | 任何配对任务 |

### 5.4 显存估算

3D SwinIR 显存消耗显著高于 2D，以 `embed_dim=48` 为例：

| H_size | window_size | batch_size | 近似显存 |
|--------|-------------|:----------:|---------|
| `[8, 32, 32]` | `[2, 4, 4]` | 1 | ~4 GB |
| `[16, 64, 64]` | `[2, 4, 4]` | 1 | ~10 GB |
| `[32, 64, 64]` | `[2, 8, 8]` | 1 | ~18 GB |
| `[32, 128, 128]` | `[4, 8, 8]` | 1 | ~35 GB |

**节省显存技巧：**
1. 减小 `H_size`（特别是 D 方向）
2. 减小 `embed_dim`（48 → 32）
3. 设置 `use_checkpoint: true`（用计算换显存）
4. 减小 `window_size`

---

## 6. 训练命令

```bash
# 各向同性 3D SR（三个维度同时 x2）
python main_train_psnr.py --opt options/train_swinir3d_sr.json

# 厚层 CT 面内超分（Z 不变，XY x2）
python main_train_psnr.py --opt options/train_swinir3d_sr.json
# 将 scale 和 upscale 改为 [1, 2, 2], dataset_type 改为 "sr3d_aniso_xy"

# 分布式训练（4 GPUs）
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 \
    main_train_psnr.py --opt options/train_swinir3d_sr.json --dist True
```

---

## 7. 快速验证脚本

训练前，用此脚本验证数据管线是否正常：

```python
import sys, torch
from torch.utils.data import DataLoader

# 验证 Dataset
sys.path.insert(0, '.')
from data.dataset_3d import Dataset3D

opt = {
    'dataroot_H': 'trainsets/trainH_3d',
    'dataroot_L': None,
    'dataset_type': 'sr3d',
    'n_channels': 1,
    'scale': (2, 2, 2),
    'H_size': (16, 32, 32),
    'phase': 'train',
    'augment': True,
}

dataset = Dataset3D(opt)
print(f"数据集大小: {len(dataset)}")
assert len(dataset) > 0, "数据集为空！"

# 验证一个 batch
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
batch = next(iter(loader))
print(f"LR: {batch['L'].shape}")  # 期望: (1, 1, 8, 16, 16)
print(f"HR: {batch['H'].shape}")  # 期望: (1, 1, 16, 32, 32)

# 验证模型
from models.network_swinir_3d import SwinIR3D
model = SwinIR3D(
    img_size=opt['H_size'], in_chans=1, embed_dim=48,
    depths=(2, 2), num_heads=(4, 4), window_size=(2, 4, 4),
    upscale=opt['scale'], upsampler='pixelshuffle3d', img_range=1.0
)
y = model(batch['L'])
print(f"输出: {y.shape}")  # 期望: (1, 1, 16, 32, 32)
print("\n✓ 数据管线验证通过！")
```
