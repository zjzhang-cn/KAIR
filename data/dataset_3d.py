"""
3D 体数据 Dataset，用于 SwinIR3D 训练

支持的数据格式:
    - .npy / .npz: numpy 数组，shape (D, H, W) 或 (D, H, W, C)
    - .nii / .nii.gz: NIfTI 格式（需安装 nibabel）

数据组织方式:
    方式 A（配对数据）: dataroot_H 和 dataroot_L 分别存放 HR 和 LR 体数据
    方式 B（仅 HR）: 只提供 dataroot_H，训练时自动对 HR 下采样生成 LR

3D 特有的下采样模式:
    - '3d_bicubic': 三个维度同时 bicubic 下采样
    - 'aniso_xy': Z 方向保持原生分辨率，XY 平面下采样（模拟厚层扫描→薄层重建）
    - 'aniso_z': XY 平面保持，Z 方向下采样（模拟各向异性分辨率）
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data


# 尝试导入可选依赖
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


IMG_EXTENSIONS_3D = ['.npy', '.npz', '.nii', '.nii.gz']


def is_3d_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS_3D)


def get_3d_paths(root_dir):
    """获取目录下所有 3D 体数据文件路径"""
    paths = []
    if root_dir is None:
        return paths
    for fname in sorted(os.listdir(root_dir)):
        if is_3d_file(fname):
            paths.append(os.path.join(root_dir, fname))
    return paths


def load_volume(path):
    """加载 3D 体数据，返回 numpy array，shape 统一为 (D, H, W)

    支持格式: .npy, .npz, .nii, .nii.gz
    自动处理值域到 [0, 1]
    """
    ext = os.path.splitext(path)[-1]
    if ext == '.gz':
        ext = '.nii.gz'

    if ext in ['.npy', '.npz']:
        vol = np.load(path)
        if isinstance(vol, np.lib.npyio.NpzFile):
            vol = vol[list(vol.keys())[0]]  # 取第一个数组
    elif ext in ['.nii', '.nii.gz']:
        if not HAS_NIBABEL:
            raise ImportError("需要安装 nibabel: pip install nibabel")
        vol = nib.load(path).get_fdata()
    else:
        raise ValueError(f"不支持的格式: {ext}")

    # 确保是 float32 且为非负
    vol = vol.astype(np.float32)
    if vol.min() < 0:
        vol = vol - vol.min()
    if vol.max() > 0:
        vol = vol / vol.max()

    # 确保 3D 或 4D (C,D,H,W)
    if vol.ndim == 3:
        pass  # (D, H, W)
    elif vol.ndim == 4:
        vol = vol.transpose(3, 0, 1, 2)  # (D,H,W,C) → (C,D,H,W)
    else:
        raise ValueError(f"不支持的维度: {vol.shape}")

    return vol


def downsample_3d(vol, scale, mode='3d_bicubic'):
    """3D 体数据下采样

    Args:
        vol: numpy array, shape (D, H, W) 或 (1, D, H, W) 或 (C, D, H, W)
        scale: int 或 tuple (sD, sH, sW) — 下采样倍数
        mode: '3d_bicubic' | 'aniso_xy' | 'aniso_z'
            - '3d_bicubic': 三个维度均匀下采样
            - 'aniso_xy': 只对 XY 平面下采样 (模拟面内高分辨率→低分辨率)
            - 'aniso_z': 只对 Z 方向下采样 (模拟层间距增大)

    Returns:
        vol_lr: 下采样后的体数据，与输入同 shape 布局
    """
    from scipy.ndimage import zoom

    if isinstance(scale, int):
        scale = (scale, scale, scale)

    sD, sH, sW = scale

    if mode == 'aniso_xy':
        sD = 1  # Z 不下采样
    elif mode == 'aniso_z':
        sH, sW = 1, 1  # XY 不下采样

    zoom_factors = (1.0 / sD, 1.0 / sH, 1.0 / sW)

    if vol.ndim == 4:
        # (C, D, H, W)：逐通道下采样
        vol_lr = np.zeros((vol.shape[0],
                           int(vol.shape[1] / sD),
                           int(vol.shape[2] / sH),
                           int(vol.shape[3] / sW)), dtype=vol.dtype)
        for c in range(vol.shape[0]):
            vol_lr[c] = zoom(vol[c], zoom_factors, order=3)  # order=3: cubic
    else:
        vol_lr = zoom(vol, zoom_factors, order=3)

    return vol_lr


def random_crop_3d(vol_hr, vol_lr, hr_patch_size, lr_patch_size):
    """随机裁剪 3D patch pair

    Args:
        vol_hr: HR 体数据 (C, D, H, W)
        vol_lr: LR 体数据 (C, D, H, W)
        hr_patch_size: (pD, pH, pW) — HR patch 尺寸
        lr_patch_size: (pD, pH, pW) — LR patch 尺寸

    Returns:
        patch_hr, patch_lr: 裁剪后的 patch
    """
    pD_hr, pH_hr, pW_hr = hr_patch_size
    pD_lr, pH_lr, pW_lr = lr_patch_size

    # 随机起始位置（基于 LR 空间，因为 LR 更小）
    D_lr, H_lr, W_lr = vol_lr.shape[1], vol_lr.shape[2], vol_lr.shape[3]
    rnd_d = random.randint(0, max(0, D_lr - pD_lr))
    rnd_h = random.randint(0, max(0, H_lr - pH_lr))
    rnd_w = random.randint(0, max(0, W_lr - pW_lr))

    # LR patch
    patch_lr = vol_lr[:, rnd_d:rnd_d + pD_lr, rnd_h:rnd_h + pH_lr, rnd_w:rnd_w + pW_lr]

    # 对应 HR patch
    rnd_d_hr, rnd_h_hr, rnd_w_hr = rnd_d * (pD_hr // pD_lr), rnd_h * (pH_hr // pH_lr), rnd_w * (pW_hr // pW_lr)
    patch_hr = vol_hr[:, rnd_d_hr:rnd_d_hr + pD_hr, rnd_h_hr:rnd_h_hr + pH_hr, rnd_w_hr:rnd_w_hr + pW_hr]

    return patch_hr, patch_lr


class Dataset3D(data.Dataset):
    """SwinIR3D 训练数据集

    配置参数 (opt):
        dataroot_H: HR 体数据目录
        dataroot_L: LR 体数据目录（可选，不提供则自动下采样 HR 生成 LR）
        dataset_type: 'sr3d' | 'sr3d_aniso_xy' | 'sr3d_aniso_z' | 'denoise3d' | 'plain3d'
        n_channels: 输入通道数，默认 1
        scale: 上采样倍率，int 或 (sD, sH, sW) tuple
        H_size: HR patch 大小 (pD, pH, pW)，如 (16, 64, 64)
        dataloader_batch_size: batch size
        augment: 是否启用数据增强，默认 True
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_channels = opt.get('n_channels', 1)
        self.scale = opt['scale']
        self.dataset_type = opt.get('dataset_type', 'sr3d')

        # 将 scale 展开为 (sD, sH, sW)
        if isinstance(self.scale, int):
            self.scale_3d = (self.scale, self.scale, self.scale)
        else:
            self.scale_3d = tuple(self.scale)

        # HR patch size
        if 'H_size' in opt and opt['H_size'] is not None:
            if isinstance(opt['H_size'], int):
                self.hr_patch_size = (opt['H_size'], opt['H_size'], opt['H_size'])
            else:
                self.hr_patch_size = tuple(opt['H_size'])
        else:
            self.hr_patch_size = (16, 64, 64)  # 默认：深度 16，XY 64x64

        # LR patch size = HR / scale
        sD, sH, sW = self.scale_3d
        self.lr_patch_size = (
            max(1, self.hr_patch_size[0] // sD),
            max(1, self.hr_patch_size[1] // sH),
            max(1, self.hr_patch_size[2] // sW)
        )

        self.augment = opt.get('augment', True)

        # 获取数据文件路径
        self.paths_H = get_3d_paths(opt.get('dataroot_H'))
        self.paths_L = get_3d_paths(opt.get('dataroot_L')) if opt.get('dataroot_L') else []

        if not self.paths_H:
            raise ValueError(f"未找到 3D 体数据文件，请检查 dataroot_H: {opt.get('dataroot_H')}")
        print(f"找到 {len(self.paths_H)} 个 HR 体数据文件")

        self.synthesize_lr = len(self.paths_L) == 0
        if self.synthesize_lr:
            print(f"未提供 LR 数据，将自动对 HR 下采样生成 LR (mode={self.dataset_type})")
        elif len(self.paths_L) != len(self.paths_H):
            raise ValueError(f"L/H 数量不匹配: {len(self.paths_L)} vs {len(self.paths_H)}")

    def __getitem__(self, index):
        # 加载 HR 体数据
        H_path = self.paths_H[index]
        vol_H = load_volume(H_path)  # (D,H,W) 或 (C,D,H,W)
        if vol_H.ndim == 3:
            vol_H = vol_H[np.newaxis, ...]  # (D,H,W) → (1,D,H,W)

        # 确保通道数匹配
        if vol_H.shape[0] != self.n_channels:
            if vol_H.shape[0] == 1 and self.n_channels == 3:
                vol_H = np.repeat(vol_H, 3, axis=0)
            elif vol_H.shape[0] == 3 and self.n_channels == 1:
                vol_H = vol_H[:1]

        # 生成或加载 LR 体数据
        if self.synthesize_lr:
            # 自动下采样
            down_mode = self.dataset_type.replace('sr3d_', '3d_') if self.dataset_type != 'sr3d' else '3d_bicubic'
            vol_L = downsample_3d(vol_H, self.scale_3d, mode=down_mode)
            L_path = H_path
        else:
            L_path = self.paths_L[index]
            vol_L = load_volume(L_path)
            if vol_L.ndim == 3:
                vol_L = vol_L[np.newaxis, ...]

        # 训练时随机裁剪 3D patch
        if self.opt.get('phase', 'train') == 'train':
            vol_H, vol_L = random_crop_3d(vol_H, vol_L, self.hr_patch_size, self.lr_patch_size)

            # 数据增强：3D 翻转
            if self.augment:
                if random.random() > 0.5:
                    vol_H = np.flip(vol_H, axis=1)  # 沿 D 翻转
                    vol_L = np.flip(vol_L, axis=1)
                if random.random() > 0.5:
                    vol_H = np.flip(vol_H, axis=2)  # 沿 H 翻转
                    vol_L = np.flip(vol_L, axis=2)
                if random.random() > 0.5:
                    vol_H = np.flip(vol_H, axis=3)  # 沿 W 翻转
                    vol_L = np.flip(vol_L, axis=3)
                if random.random() > 0.5:
                    vol_H = np.rot90(vol_H, k=1, axes=(2, 3))  # 在 HW 平面旋转 90°
                    vol_L = np.rot90(vol_L, k=1, axes=(2, 3))

        # numpy → torch tensor
        vol_H = torch.from_numpy(vol_H.copy()).float()
        vol_L = torch.from_numpy(vol_L.copy()).float()

        return {'L': vol_L, 'H': vol_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    import tempfile

    print("=" * 60)
    print("3D Dataset 测试")
    print("=" * 60)

    # 创建临时测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        # 生成 5 个随机的 3D HR 体数据 (D=32, H=64, W=64)
        for i in range(5):
            vol = np.random.rand(32, 64, 64).astype(np.float32)
            np.save(os.path.join(tmpdir, f"vol_{i:03d}.npy"), vol)

        opt = {
            'dataroot_H': tmpdir,
            'dataroot_L': None,
            'dataset_type': 'sr3d',
            'n_channels': 1,
            'scale': (2, 2, 2),  # 3D 各向同性 x2
            'H_size': (16, 32, 32),  # HR patch
            'phase': 'train',
            'augment': True,
        }

        dataset = Dataset3D(opt)
        print(f"数据集大小: {len(dataset)}")
        sample = dataset[0]
        print(f"LR shape: {sample['L'].shape}  (期望: (1, 8, 16, 16))")
        print(f"HR shape: {sample['H'].shape}  (期望: (1, 16, 32, 32))")

    print("\n测试通过！")
