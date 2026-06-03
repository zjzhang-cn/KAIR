# SwinIR 详解及各向异性超分修改

## 概述

**SwinIR**（Swin Transformer for Image Restoration）是 Jingyun Liang 等人在 ICCV 2021 上提出的图像复原模型，将 Swin Transformer 引入图像复原领域。相比传统 CNN（如 DnCNN、MSRResNet），SwinIR 利用 Transformer 的自注意力机制捕捉长距离像素依赖，同时通过窗口划分和窗口平移大幅降低计算量。

**论文**: [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257)

**代码位置**：
- 网络定义：`models/network_swinir.py`
- 测试脚本：`main_test_swinir.py`
- 训练配置：`options/swinir/`

---

## 网络结构

SwinIR 采用三段式结构：

```
输入 L (低质量图)
    │
    ▼
┌──────────────────────────────────────┐
│ 1. 浅层特征提取 (shallow feature)      │
│    conv_first: Conv2d(3→embed_dim)   │
│    → 提取底层边缘/纹理特征              │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ 2. 深层特征提取 (deep feature)         │
│    ┌─────────────────────────────┐   │
│    │ RSTB × N (残差Swin Transformer块) │   │
│    │  每个 RSTB 内含:                 │   │
│    │  - BasicLayer (多个 Swin Block)  │   │
│    │    ├─ W-MSA  (窗口多头自注意力)    │   │
│    │    └─ SW-MSA (平移窗口自注意力)    │   │
│    │  - 3×3 Conv (残差连接 + 归纳偏置) │   │
│    └─────────────────────────────┘   │
│    conv_after_body → + 残差连接        │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ 3. 高质量图像重建 (HQ reconstruction)   │
│    - 去噪/去压缩：直接 conv_last 输出    │
│    - 经典 SR：PixelShuffle 上采样        │
│    - 轻量 SR：PixelShuffle 一步上采样     │
│    - 真实 SR：最近邻插值 + 卷积           │
│    - 各向异性SR：F.interpolate + 卷积     │
└──────────────────────────────────────┘
    │
    ▼
输出 E (高质量复原图)
```

---

## 关键技术

### 1. 窗口多头自注意力 (W-MSA)

传统 Transformer 的自注意力计算量与图像像素数的平方成正比。SwinIR 将图像**划分为固定大小的窗口**（默认 8×8），只在窗口内计算注意力，将复杂度从 `O(H²W²)` 降到 `O(window_size² × HW)`。

### 2. 平移窗口 (SW-MSA)

W-MSA 只在窗口内计算，窗口之间没有信息流通。SW-MSA 将窗口**平移半个窗口大小**后再划分，使得相邻窗口的像素可以交互，建立跨窗口连接。

偶数层使用 W-MSA，奇数层使用 SW-MSA：

```python
shift_size = 0 if (i % 2 == 0) else window_size // 2
```

### 3. RSTB (Residual Swin Transformer Block)

每个 RSTB = 一组 Swin Transformer Block + 残差卷积连接：

```python
return self.patch_embed(
    self.conv(
        self.patch_unembed(self.residual_group(x, x_size), x_size)
    )
) + x  # 残差连接
```

这里的 `self.conv`（1×1 或 3×3 卷积）提供额外的卷积归纳偏置（局部性先验），有助于稳定训练。

### 4. 相对位置编码

不使用绝对位置编码，而是用**可学习的相对位置偏置**（Relative Position Bias），加在注意力矩阵上：

```python
attn = attn + relative_position_bias.unsqueeze(0)
```

相对位置偏置表大小：`(2*Wh-1) × (2*Ww-1) × num_heads`，通过在注意力分数上加上偏置来编码窗口内 token 之间的相对位置关系。

---

## 支持的 6 种任务

| 任务 | `--task` | upscale | 上采样器 | 输入通道 | 用途 |
|------|----------|---------|----------|---------|------|
| 经典图像 SR | `classical_sr` | 2/3/4/8 | `pixelshuffle` | 3 | 标准超分（如 bicubic 下采样） |
| 轻量图像 SR | `lightweight_sr` | 2/3/4 | `pixelshuffledirect` | 3 | 参数更少的超分 |
| 真实图像 SR | `real_sr` | 4 | `nearest+conv` | 3 | 真实世界退化图像超分 |
| 灰度去噪 | `gray_dn` | 1 | 无 | 1 | 灰度图像去噪 |
| 彩色去噪 | `color_dn` | 1 | 无 | 3 | 彩色图像去噪 |
| JPEG 去压缩 | `jpeg_car` | 1 | 无 | 1 | 去除 JPEG 压缩伪影 |

### 典型模型参数量

| 任务 | 配置 | 参数量 |
|------|------|--------|
| 经典 SR | embed_dim=180, depths=[6,6,6,6,6,6] | ~11.5M |
| 轻量 SR | embed_dim=60, depths=[6,6,6,6] | ~0.9M |
| 真实 SR (大) | embed_dim=240, depths=[9×6] | ~28M |
| 去噪/去压缩 | embed_dim=180, depths=[6,6,6,6,6,6] | ~11.5M |

---

## 测试命令

```bash
# 经典 SR x4
python main_test_swinir.py --task classical_sr --scale 4 \
  --model_path model_zoo/swinir_x4.pth \
  --folder_lq testsets/set5/LR --folder_gt testsets/set5/HR

# 彩色去噪，噪声等级 25
python main_test_swinir.py --task color_dn --noise 25 \
  --model_path model_zoo/swinir_color_dn25.pth \
  --folder_gt testsets/bsd68

# 真实世界 SR
python main_test_swinir.py --task real_sr \
  --model_path model_zoo/swinir_real_sr_x4.pth \
  --folder_lq path/to/low_quality_images

# 大图分块推理（避免显存溢出）
python main_test_swinir.py --task real_sr --tile 512 --tile_overlap 32 \
  --model_path model_zoo/swinir_real_sr_x4.pth \
  --folder_lq path/to/images
```

---

## 训练配置

训练 JSON 位于 `options/swinir/`：

| 文件 | 对应任务 |
|------|---------|
| `train_swinir_sr_classical.json` | 经典图像 SR |
| `train_swinir_sr_lightweight.json` | 轻量图像 SR |
| `train_swinir_sr_realworld_x4_psnr.json` | 真实 SR (PSNR) |
| `train_swinir_sr_realworld_x4_gan.json` | 真实 SR (GAN) |
| `train_swinir_denoising_gray.json` | 灰度去噪 |
| `train_swinir_denoising_color.json` | 彩色去噪 |
| `train_swinir_car_jpeg.json` | JPEG 去压缩 |

训练时 `net_type` 设置为 `"swinir"`，通过 `select_network.py` 工厂创建网络，与 `main_train_psnr.py` 共用训练框架。

```bash
python main_train_psnr.py --opt options/swinir/train_swinir_sr_classical.json
```

---

## 各向异性超分修改

### 背景

原始的 SwinIR 上采样模块（`PixelShuffle`）是各向同性的——高和宽使用相同的放大倍数。在实际场景中，可能需要只在一个方向上做超分，例如：

- 将横向压缩过的图像恢复宽度，高度不变
- 将竖向压缩过的图像恢复高度，宽度不变
- 高和宽以不同倍率放大

### 修改内容

在 `models/network_swinir.py` 的 `SwinIR` 类中新增以下能力：

#### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `upscale_h` | int | 等于 `upscale` | 高度方向放大倍数 |
| `upscale_w` | int | 等于 `upscale` | 宽度方向放大倍数 |

默认值等于原来的 `upscale`，保证完全向后兼容。

#### 新增上采样模式 `'aniso'`

使用 `F.interpolate(scale_factor=(h, w), mode='nearest')` 直接指定 H 和 W 各自的上采样倍率，打破 `PixelShuffle` 的方形限制。插值后跟两层卷积（`conv_up` + `conv_hr`）对插值结果做细化处理。

```
conv_before_upsample → F.interpolate(scale=(h,w)) → conv_up → conv_hr → conv_last
```

#### 网络初始化代码

```python
# 只放大宽度 x4，高度不变（图像变扁）
model = SwinIR(
    upscale=4,              # 保留原参数以兼容
    upscale_h=1,            # H 不变
    upscale_w=4,            # W 放大 4x
    img_size=(64, 64),
    window_size=8,
    img_range=1.,
    depths=[6, 6, 6, 6],
    embed_dim=60,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='aniso',      # 使用各向异性模式
    resi_connection='1conv'
)

# 输入:  (1, 3, 64, 64)   H=64, W=64
# 输出:  (1, 3, 64, 256)  H=64, W=256
```

#### 常用场景

| 场景 | `upscale_h` | `upscale_w` | 效果 |
|------|:----------:|:----------:|------|
| 只扩宽度 | 1 | 4 | 图像变扁，宽度拉伸 4x |
| 只扩高度 | 4 | 1 | 图像变窄，高度拉伸 4x |
| 非对称放大 | 2 | 4 | H 放大 2x，W 放大 4x |
| H 不动 W 扩 | 1 | 3 | 只恢复横向分辨率 |
| 正常 SR | 4 | 4 | 等同于原始的 `upscale=4` |

### 关键代码变更

**1. 新增参数（`__init__`）**

```python
def __init__(self, ..., upscale=2, upscale_h=None, upscale_w=None, ...):
    self.upscale_h = upscale_h if upscale_h is not None else upscale
    self.upscale_w = upscale_w if upscale_w is not None else upscale
    self.is_aniso = (self.upscale_h != self.upscale_w)
```

**2. 新增 `'aniso'` 上采样模块**

```python
elif self.upsampler == 'aniso':
    self.conv_before_upsample = nn.Sequential(
        nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
        nn.LeakyReLU(inplace=True))
    self.conv_up = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
    self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
```

**3. 前向传播中的各向异性上采样**

```python
elif self.upsampler == 'aniso':
    x = self.conv_first(x)
    x = self.conv_after_body(self.forward_features(x)) + x
    x = self.conv_before_upsample(x)
    x = self.lrelu(self.conv_up(
        F.interpolate(x, scale_factor=(self.upscale_h, self.upscale_w),
                       mode='nearest')))
    x = self.conv_last(self.lrelu(self.conv_hr(x)))
```

**4. 输出裁剪适配各向异性**

```python
return x[:, :, :H*self.upscale_h, :W*self.upscale_w]
```

### 验证结果

```
=== 测试 1: 只放大宽度 x4，高度不变 ===
输入: (1, 3, 64, 64)  →  输出: (1, 3, 64, 256)  ✓

=== 测试 2: H x2, W x4 ===
输入: (1, 3, 48, 64)  →  输出: (1, 3, 96, 256)  ✓

=== 测试 3: 默认各向同性 (经典 SR x4) ===
输入: (1, 3, 64, 64)  →  输出: (1, 3, 256, 256)  ✓
```
