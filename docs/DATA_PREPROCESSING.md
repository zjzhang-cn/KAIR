# 训练数据预处理说明

## 1. 图像读取与格式转换

所有 dataset 的数据入口从 `imread_uint()` 开始（`utils/utils_image.py:189`）：

- `cv2.imread(img_path, cv2.IMREAD_UNCHANGED)` 读取，返回 uint8 HWC numpy 数组
- 将 BGR 转为 RGB（`cv2.COLOR_BGR2RGB`）
- 灰度图自动扩展为 HxWx1

### 数值范围转换链

```
uint8 [0,255] —uint2tensor3()→ tensor float [0,1] CHW  (训练时)
uint8 [0,255] —uint2single()→ numpy float32 [0,1] HWC  (测试时) —single2tensor3()→ tensor float CHW
```

关键函数（`utils/utils_image.py`）：

| 函数 | 输入 | 输出 | 注 |
|------|------|------|-----|
| `imread_uint(path, n_channels)` | 路径 | numpy uint8 HWC | 读取 |
| `uint2single(img)` | numpy uint8 | numpy float32 [0,1] | `img/255.0` |
| `uint2tensor3(img)` | numpy uint8 | tensor float [0,1] CHW | 一次到位 |
| `single2tensor3(img)` | numpy float32 [0,1] | tensor float CHW | 不除 255 |
| `tensor2uint(img)` | tensor | numpy uint8 HWC | 反向 |

---

## 2. 数据增强

### 2.1 翻转和旋转（唯一增强方式）

`utils/utils_image.py:384` `augment_img(img, mode=0~7)`：

| mode | 操作 |
|------|------|
| 0 | 不变 |
| 1 | 翻转上 + 旋转 90° |
| 2 | 水平翻转(上下) |
| 3 | 旋转 270° |
| 4 | 翻转上 + 旋转 180° |
| 5 | 旋转 90° |
| 6 | 旋转 180° |
| 7 | 翻转上 + 旋转 270° |

训练时随机取 mode(0-7)，L 和 H 做同样的变换保持一致：
```python
mode = random.randint(0, 7)
img_L, img_H = augment_img(img_L, mode=mode), augment_img(img_H, mode=mode)
```

### 2.2 3D 体数据增强（CT/MR）

`data/dataset_3d.py:253-266`：仅沿 D/H/W 轴随机翻转 + HW 平面 90° 旋转。

### 2.3 没有的增强

- 没有随机仿射变换（平移、缩放、剪切）
- 没有弹性形变（ElasticDeformation）
- 没有随机透视变换
- 没有颜色扰动（亮度、对比度、饱和度）

---

## 3. 各任务数据集差异

### 3.1 超分辨率 SR（`dataset_sr.py`）

```
读取 H 图(uint8) → modcrop(裁到能被 scale 整除) → uint2single → bicubic 下采样生成 L
→ 随机裁剪 L_patch(L_size) + 对应 H_patch(H_size=L_size*scale) → augment_img → uint2tensor3
```

关键点：
- `modcrop()`（`utils/utils_image.py:498`）裁掉边缘像素使 H/W 能被 scale 整除
- L 由 H 经 bi-cubic 下采样生成（`utils/utils_image.py:925` `imresize_np`，MATLAB 风格 cubic）
- L 和 H 裁剪位置对应：`rnd_h_H = rnd_h * scale`

### 3.2 去噪 DnCNN（`dataset_dncnn.py`）

```
读取 H 图(uint8) → 随机裁剪 patch → augment_img → uint2tensor3 → 加高斯噪声 → L
```

噪声公式：`noise = torch.randn(size) * (sigma / 255.0)`，sigma 固定（如 15/25/50）

### 3.3 去噪 FFDNet / FDnCNN（`dataset_ffdnet.py` / `dataset_fdncnn.py`）

```
读取 H 图(uint8) → 随机裁剪 patch → augment_img → uint2tensor3 → 加高斯噪声 → L
```

区别：sigma **在 `[sigma_min, sigma_max]` 范围内均匀随机采样**：
```python
noise_level = torch.FloatTensor([np.random.uniform(sigma_min, sigma_max)]) / 255.0
noise = torch.randn(img_L.size()).mul_(noise_level).float()
```
- FFDNet：`noise_level` 作为额外输入 `C` 传给网络
- FDnCNN：将 `noise_level` 拼成全 1 map 拼入 `L` 的第 4 通道

### 3.4 去噪 DPSR / SRMD（`dataset_dpsr.py` / `dataset_srmd.py`）

与 FFDNet 类似，sigma 范围随机，但额外有 10% 概率 noise_level=0（不加噪）：
```python
if random.random() < 0.1:
    noise_level = torch.zeros(1).float()
else:
    noise_level = torch.FloatTensor([np.random.uniform(sigma_min, sigma_max)]) / 255.0
```

### 3.5 通用配对数据集（`dataset_plain.py`）

```
读取 H 图 + L 图（都必须提供）→ 随机裁剪 patch → augment_img → uint2tensor3
```
最简单，不做任何退化合成。

---

## 4. 测试阶段的区别

- sigma 使用配置中的 `sigma_test` 固定值（而非训练时的随机范围）
- 噪声通过 `np.random.normal(0, sigma_test/255.0, shape)` 在 numpy 上生成
- `np.random.seed(seed=0)` 固定种子保证可复现
- 不裁剪 patch，整图推理

---

## 5. 随机噪声生成汇总

| 数据集 | sigma 策略 | 噪声生成 | noise level 输入 |
|--------|-----------|---------|-----------------|
| DnCNN | 固定值（配置 `sigma`） | `torch.randn * sigma/255` | 无 |
| FFDNet | 随机范围 `[min, max]` | `torch.randn * uniform(min,max)/255` | `C` 通道 |
| FDnCNN | 随机范围 `[min, max]` | 同上 | noise level map 拼入 L |
| DPSR | 随机范围，10% 为 0 | 同上 | 有 |
| SRMD | 随机范围，10% 为 0 | 同上 | 有 |
| SR | 无噪声 | bicubic 下采样 | 无 |

---

## 6. 数据准备建议

- 训练数据只需准备干净的高质量图像（`dataroot_H`），低质量图由数据集在线合成
- 可选：用 `split_imageset()`（`utils/utils_image.py:123`）将大图预切成小 patch 加速 IO
- 3D 体数据支持 `.npy/.npz` 和 `.nii/.nii.gz`（NIfTI 需安装 `nibabel`）
- 3D 下采样模式：`3d_bicubic`（各向同性）、`aniso_xy`（仅 XY 平面）、`aniso_z`（仅 Z 方向）
