# -----------------------------------------------------------------------------------
# SwinIR3D: 3D Volumetric Image Restoration Using 3D Swin Transformer
# 基于 SwinIR (https://arxiv.org/abs/2108.10257) 改造为 3D 版本
# 适用于医学 CT/MRI 体数据超分、3D 去噪等任务
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_


# ============================================================================
# 基础组件
# ============================================================================

class Mlp(nn.Module):
    """3D 版本 MLP（与 2D 相同，因为操作在 channel 维度）"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_3d(x, window_size):
    """
    3D 窗口划分：将 3D 体数据划分为不重叠的 3D 窗口
    Args:
        x: (B, D, H, W, C)  5D 特征图
        window_size: (wD, wH, wW)  3D 窗口大小
    Returns:
        windows: (B*num_windows, wD*wH*wW, C)
    """
    B, D, H, W, C = x.shape
    wD, wH, wW = window_size
    # 重塑为网格状窗口
    x = x.view(B, D // wD, wD, H // wH, wH, W // wW, wW, C)
    # 将窗口维度移到前面，合并 batch 和窗口
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wD * wH * wW, C)
    return windows


def window_reverse_3d(windows, window_size, B, D, H, W):
    """
    3D 窗口还原：将窗口化特征还原为原始 3D 形状
    Args:
        windows: (B*num_windows, wD*wH*wW, C)
        window_size: (wD, wH, wW)
    Returns:
        x: (B, D, H, W, C)
    """
    wD, wH, wW = window_size
    x = windows.view(B, D // wD, H // wH, W // wW, wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


# ============================================================================
# 3D 窗口多头自注意力 (3D W-MSA)
# ============================================================================

class WindowAttention3D(nn.Module):
    """3D 窗口多头自注意力，带 3D 相对位置偏置

    Args:
        dim (int): 输入通道数
        window_size (tuple[int]): 3D 窗口大小 (wD, wH, wW)
        num_heads (int): 注意力头数
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wD, wH, wW)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 3D 相对位置偏置表
        # 大小为 (2*wD-1) * (2*wH-1) * (2*wW-1), num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                num_heads
            )
        )

        # 计算窗口内每对 token 的相对位置索引
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, wD, wH, wW
        coords_flatten = torch.flatten(coords, 1)  # 3, wD*wH*wW
        # 相对坐标: (3, N, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 3
        # 偏移到非负范围
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        # 将 3D 坐标编码为 1D 索引
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B*num_windows, wD*wH*wW, C)
            mask: 3D 注意力 mask, shape (num_windows, wD*wH*wW, wD*wH*wW) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, num_heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        # 3D 相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)  # N, N, num_heads
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================================
# 3D Swin Transformer Block
# ============================================================================

class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer Block: 包含 W-MSA 或 SW-MSA

    Args:
        dim (int): 输入通道数
        input_resolution (tuple[int]): 输入体数据尺寸 (D, H, W)
        num_heads (int): 注意力头数
        window_size (tuple[int]): 3D 窗口大小 (wD, wH, wW)
        shift_size (tuple[int]): 3D 平移大小 (sD, sH, sW), 0 表示 W-MSA, >0 为 SW-MSA
        mlp_ratio (float): MLP 隐层维度倍率
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=(2, 7, 7),
                 shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 如果输入尺寸小于窗口尺寸，缩小窗口并禁止平移
        for i in range(3):
            if min(self.input_resolution[i], self.input_resolution[i]) <= self.window_size[i]:
                # 不需要平移
                self.shift_size = list(self.shift_size)
                self.shift_size[i] = 0
                self.window_size = list(self.window_size)
                self.window_size[i] = min(self.input_resolution[i], self.window_size[i])
                self.shift_size = tuple(self.shift_size)
                self.window_size = tuple(self.window_size)

        assert all(0 <= s < w for s, w in zip(self.shift_size, self.window_size)), \
            "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        """前向传播

        Args:
            x: (B, D*H*W, C)
            mask_matrix: 3D 注意力 mask 矩阵
        """
        B, L, C = x.shape
        D, H, W = self.input_resolution
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # 3D 循环平移（用于 SW-MSA）
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # 3D 窗口划分
        x_windows = window_partition_3d(shifted_x, self.window_size)  # nW*B, wD*wH*wW, C

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=mask_matrix)

        # 3D 窗口还原
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1],
                                          self.window_size[2], C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, B, D, H, W)  # B D H W C

        # 反向 3D 循环平移
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, D * H * W, C)

        # FFN (前馈网络)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# ============================================================================
# 3D Patch 嵌入 / 反嵌入
# ============================================================================

class PatchEmbed3D(nn.Module):
    """3D 特征图 → 序列：仅做维度重排，不做卷积投影（特征提取已由 conv_first 完成）"""
    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """x: (B, C, D, H, W) → (B, D*H*W, C)"""
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, D*H*W, C
        if self.norm is not None:
            x = self.norm(x)
        return x, (D, H, W)


class PatchUnEmbed3D(nn.Module):
    """序列 → 3D 特征图：仅做维度重排"""
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """x: (B, D*H*W, C) → (B, C, D, H, W)"""
        B, L, C = x.shape
        D, H, W = x_size
        assert L == D * H * W, "input feature has wrong size"
        x = x.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        return x


# ============================================================================
# 3D 上采样模块
# ============================================================================

class Upsample3D(nn.Module):
    """3D PixelShuffle 上采样：支持各向同性或各向异性 3D 上采样

    使用 Conv3d + 维度重排实现 3D PixelShuffle
    scale: (sD, sH, sW) 或 int（各向同性时转为 (s, s, s)）
    """
    def __init__(self, scale, num_feat):
        super().__init__()
        if isinstance(scale, int):
            self.scale = (scale, scale, scale)
        else:
            self.scale = tuple(scale)
        sD, sH, sW = self.scale
        self.num_feat = num_feat

        # 使用 Conv3d 扩展通道数 (C → C * sD * sH * sW)，然后 reshape 实现 PixelShuffle
        self.conv = nn.Conv3d(num_feat, num_feat * sD * sH * sW, kernel_size=3, padding=1)

    def forward(self, x):
        """x: (B, C, D, H, W) → (B, C, D*sD, H*sH, W*sW)"""
        B, C, D, H, W = x.shape
        sD, sH, sW = self.scale
        x = self.conv(x)  # B, C*sD*sH*sW, D, H, W
        # reshape: 将多出的通道重新排列为空间维度
        x = x.reshape(B, C, sD, sH, sW, D, H, W)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # B, C, D, sD, H, sH, W, sW
        x = x.reshape(B, C, D * sD, H * sH, W * sW)
        return x


class Upsample3DInterp(nn.Module):
    """3D 插值上采样（用于各向异性场景，比 PixelShuffle 更灵活）"""
    def __init__(self, scale, num_feat):
        super().__init__()
        if isinstance(scale, int):
            self.scale = (scale, scale, scale)
        else:
            self.scale = tuple(scale)
        self.conv = nn.Conv3d(num_feat, num_feat, kernel_size=3, padding=1)

    def forward(self, x):
        return F.interpolate(self.conv(x), scale_factor=self.scale, mode='trilinear', align_corners=False)


# ============================================================================
# 3D RSTB (Residual Swin Transformer Block)
# ============================================================================

class RSTB3D(nn.Module):
    """3D 残差 Swin Transformer Block：Swin Block 组 + 3D 卷积残差连接

    Args:
        dim (int): 输入通道数
        input_resolution (tuple[int]): (D, H, W)
        depth (int): Swin Block 数量
        num_heads (int): 注意力头数
        window_size (tuple[int]): 3D 窗口大小 (wD, wH, wW)
        mlp_ratio (float): MLP 倍率
        use_checkpoint (bool): 是否使用 checkpoint 节省显存
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # 3D 卷积残差连接（提供空间归纳偏置）
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, x_size):
        D, H, W = x_size
        B, L, C = x.shape

        # 计算注意力 mask（用于 SW-MSA）
        # 简化：直接为每个 block 计算 mask
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, self._get_mask(D, H, W, blk, x.device))
            else:
                x = blk(x, self._get_mask(D, H, W, blk, x.device))

        # 3D 卷积 + 残差
        x_3d = x.transpose(1, 2).view(B, C, D, H, W)
        x_3d = self.conv(x_3d) + x_3d
        x = x_3d.flatten(2).transpose(1, 2)

        return x

    def _get_mask(self, D, H, W, blk, device):
        """计算 3D SW-MSA 的注意力 mask"""
        if all(s == 0 for s in blk.shift_size):
            return None

        # 3D 循环平移 mask 计算
        wD, wH, wW = blk.window_size
        sD, sH, sW = blk.shift_size

        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        cnt = 0
        for d in (slice(None, -sD) if sD > 0 else slice(None),):
            for h in (slice(None, -sH) if sH > 0 else slice(None),):
                for w in (slice(None, -sW) if sW > 0 else slice(None),):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, (wD, wH, wW))
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


# ============================================================================
# SwinIR3D 主网络
# ============================================================================

class SwinIR3D(nn.Module):
    """SwinIR3D: 基于 3D Swin Transformer 的体数据图像复原

    适用场景:
        - 医学 CT/MRI 超分辨率 (upscale > 1)
        - 3D 体数据去噪 (upscale = 1)
        - 3D 荧光显微镜图像增强

    Args:
        img_size (tuple[int]): 输入体数据尺寸 (D, H, W)，如 (32, 64, 64)
        in_chans (int): 输入通道数（如 CT=1, RGB 视频=3）
        embed_dim (int): Patch 嵌入维数. 默认 96
        depths (tuple[int]): 每层 RSTB 中 Swin Block 的数量
        num_heads (tuple[int]): 每层注意力头数
        window_size (tuple[int]): 3D 窗口大小 (wD, wH, wW)
        mlp_ratio (float): MLP 隐层维数倍率
        upscale (int | tuple[int]): 上采样倍率，int 表示各向同性，tuple(sD,sH,sW) 表示各向异性
        upsampler: 'pixelshuffle3d' | 'interp3d' | '' (去噪任务用 '')
        img_range: 图像像素值范围，1. 或 255.
        use_checkpoint (bool): 是否使用 checkpoint 节省显存
    """
    def __init__(self, img_size=(32, 64, 64), in_chans=1,
                 embed_dim=96, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=(2, 4, 4), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2,
                 img_range=1., upsampler='pixelshuffle3d',
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64

        self.img_range = img_range
        self.upscale = upscale
        if isinstance(upscale, int):
            self.scale_tuple = (upscale, upscale, upscale)
        else:
            self.scale_tuple = tuple(upscale)
        self.upsampler = upsampler
        self.window_size = window_size

        # Patch embedding：特征图 (B,C,D,H,W) → 序列 (B,D*H*W,C)
        self.patch_embed = PatchEmbed3D(
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        # 计算 patch 后的特征图尺寸
        self.patches_resolution = img_size  # patch_size=1 时不变

        # 绝对位置编码（可选）
        self.ape = ape
        if self.ape:
            num_patches = img_size[0] * img_size[1] * img_size[2]
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth 衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建 RSTB3D 层
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = RSTB3D(
                dim=embed_dim,
                input_resolution=img_size,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim)

        # 重建模块
        if self.upsampler == 'pixelshuffle3d':
            # 3D PixelShuffle 上采样
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3D(scale=self.scale_tuple, num_feat=num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'interp3d':
            # 3D 插值上采样（更灵活）
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3DInterp(scale=self.scale_tuple, num_feat=num_feat)
            self.conv_hr = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # 去噪任务：不上采样
            self.conv_last = nn.Conv3d(embed_dim, num_out_ch, 3, 1, 1)

        # 浅层特征提取
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_image_size(self, x):
        """将输入的 D, H, W pad 到 window_size 的倍数"""
        _, _, d, h, w = x.size()
        wD, wH, wW = self.window_size
        mod_pad_d = (wD - d % wD) % wD
        mod_pad_h = (wH - h % wH) % wH
        mod_pad_w = (wW - w % wW) % wW
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x

    def forward_features(self, x):
        """深层特征提取"""
        x, x_size = self.patch_embed(x)  # (B, D*H*W, C), (D, H, W)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B, D*H*W, C
        return x, x_size

    def forward(self, x):
        """前向传播

        Args:
            x: (B, C, D, H, W)  输入 3D 体数据
        Returns:
            y: (B, C, D*sD, H*sH, W*sW)  输出 3D 体数据
        """
        D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
        x_input = x  # 保存原始输入，用于去噪残差连接

        # Pad 到 window_size 的倍数
        x = self.check_image_size(x)

        # 浅层特征提取
        x_first = self.conv_first(x)  # (B, embed_dim, D, H, W)

        # 深层特征提取（Swin Transformer）
        x_feat, feat_size = self.forward_features(x_first)  # (B, D*H*W, embed_dim)
        D, H, W = feat_size
        x_feat_3d = x_feat.transpose(1, 2).view(x_first.shape[0], -1, D, H, W)

        # 残差连接：深层特征 + 浅层特征
        x = self.conv_after_body(x_feat_3d) + x_first  # (B, embed_dim, D, H, W)

        # 重建
        if self.upsampler == 'pixelshuffle3d':
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
        elif self.upsampler == 'interp3d':
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # 去噪：不上采样，残差输出（加到原始输入上，而非加到浅层特征上）
            res = x + x_first  # conv_after_body(output) + conv_first(input)
            x = self.conv_last(res)  # embed_dim → in_chans
            x = x + x_input  # 全局残差：输入 + 预测残差

        # 裁剪到目标尺寸
        sD, sH, sW = self.scale_tuple
        x = x[:, :, :D_in * sD, :H_in * sH, :W_in * sW]

        return x


# ============================================================================
# 测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SwinIR3D 测试")
    print("=" * 60)

    # 测试 1: 3D 各向同性超分 (x2)
    print("\n=== 测试 1: 3D 各向同性 SR x2 ===")
    model = SwinIR3D(
        img_size=(32, 32, 32),  # 输入体数据尺寸 D=32, H=32, W=32
        in_chans=1,              # CT 图像单通道
        embed_dim=48,            # 小模型便于测试
        depths=(2, 2),           # 2 层 RSTB
        num_heads=(4, 4),
        window_size=(2, 4, 4),   # 3D 窗口 (D方向小因为CT层间距大)
        upscale=2,
        upsampler='pixelshuffle3d'
    )
    x = torch.randn(1, 1, 32, 32, 32)  # CT volume: B=1, C=1, D=32, H=32, W=32
    y = model(x)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"输入:  {x.shape}")
    print(f"输出:  {y.shape}")
    print(f"参数量: {num_params / 1e6:.2f}M")

    # 测试 2: 3D 各向异性超分 (只在 XY 平面放大, Z 不变)
    print("\n=== 测试 2: 3D 各向异性 SR (Z不放大, XY x2) ===")
    model2 = SwinIR3D(
        img_size=(16, 32, 32),
        in_chans=1,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(4, 4),
        window_size=(2, 4, 4),
        upscale=(1, 2, 2),       # Z不变, Hx2, Wx2
        upsampler='interp3d'
    )
    x2 = torch.randn(1, 1, 16, 32, 32)
    y2 = model2(x2)
    num_params2 = sum(p.numel() for p in model2.parameters())
    print(f"输入:  {x2.shape}")
    print(f"输出:  {y2.shape}")
    print(f"参数量: {num_params2 / 1e6:.2f}M")

    # 测试 3: 3D 去噪 (upscale=1)
    print("\n=== 测试 3: 3D 去噪 (不上采样) ===")
    model3 = SwinIR3D(
        img_size=(16, 32, 32),
        in_chans=1,
        embed_dim=48,
        depths=(2, 2),
        num_heads=(4, 4),
        window_size=(2, 4, 4),
        upscale=1,
        upsampler=''
    )
    x3 = torch.randn(1, 1, 16, 32, 32)
    y3 = model3(x3)
    print(f"输入:  {x3.shape}")
    print(f"输出:  {y3.shape} (尺寸不变)")

    print("\n所有测试通过！")
