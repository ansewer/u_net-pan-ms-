
"""
真实分辨率推理脚本：仅有 MS_up 与 PAN（无 HR-MS 真值）的泛锐化输出

用途：
- 对老师给的真实测试集目录（MS_up_800, PAN_cut_800）进行融合推理；
- 由于无 GT，不能计算 SAM/PSNR 等参考指标，因此提供 Auto-t（test-time 自适应注入强度选择）：
    在一组候选 t 上生成融合结果，用无参考自评分 score 选取“更稳”的 t。

"""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import glob
import argparse
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List

import numpy as np
import tifffile as tiff
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def try_import_wandb():
    try:
        import wandb  # type: ignore  # 类型检查忽略
        return wandb
    except Exception:
        return None


def set_seed(seed: int = 1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ----------------------------
# TIF 读写 + 归一化
# ----------------------------
def _to_chw(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4, 6, 8, 10, 12) and arr.shape[1] > 8 and arr.shape[2] > 8:
            return arr
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unsupported tif shape: {arr.shape}")


def read_tif_chw_keep_dtype(path: str) -> np.ndarray:
    return _to_chw(np.asarray(tiff.imread(path)))


def _infer_int_scale(max_val: int) -> int:
    candidates = [255, 1023, 2047, 4095, 8191, 16383, 32767, 65535]
    for c in candidates:
        if max_val <= c:
            return c
    return candidates[-1]


def normalize_01(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.integer):
        max_val = int(arr.max())
        scale = _infer_int_scale(max_val)
        return np.clip(arr.astype(np.float32) / float(scale), 0.0, 1.0)
    a = arr.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mn >= -0.05 and mx <= 1.2:
        return np.clip(a, 0.0, 1.0)
    if mn >= 0.0 and mx > 1e-6:
        return np.clip(a / mx, 0.0, 1.0)
    lo = np.percentile(a, 0.5)
    hi = np.percentile(a, 99.5)
    if hi - lo < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def pick_train_dir(data_root: str) -> str:
    cand1 = os.path.join(data_root, "train_data", "train")
    cand2 = os.path.join(data_root, "train_data")
    return cand1 if os.path.isdir(cand1) else cand2


def pick_test_dir(data_root: str) -> str:
    cand1 = os.path.join(data_root, "test_data", "test")
    cand2 = os.path.join(data_root, "test_data")
    return cand1 if os.path.isdir(cand1) else cand2


# ----------------------------
# 图像操作
# ----------------------------
def gaussian_kernel_1d(ksize: int, sigma: float, device, dtype):
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    return k / (k.sum() + 1e-12)


def gaussian_blur(x: torch.Tensor, sigma: float = 1.6, ksize: int = 11) -> torch.Tensor:
    B, C, H, W = x.shape
    k = gaussian_kernel_1d(ksize, sigma, x.device, x.dtype)
    k1 = k.view(1, 1, ksize, 1).repeat(C, 1, 1, 1)
    k2 = k.view(1, 1, 1, ksize).repeat(C, 1, 1, 1)
    x = F.conv2d(x, k1, padding=(ksize // 2, 0), groups=C)
    x = F.conv2d(x, k2, padding=(0, ksize // 2), groups=C)
    return x


def high_pass(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    return x - gaussian_blur(x, sigma=sigma, ksize=11)


def degrade_to_lr(x_hr: torch.Tensor, scale: int = 4) -> torch.Tensor:
    return F.interpolate(x_hr, scale_factor=1 / scale, mode="bicubic", align_corners=False)


def upsample_to_hr(x_lr: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x_lr, size=size_hw, mode="bicubic", align_corners=False)


def sobel_gx_gy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy


def edge_mask_from_pan(pan: torch.Tensor, power: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    gx, gy = sobel_gx_gy(pan)
    g = torch.sqrt(gx * gx + gy * gy + eps)
    B = g.shape[0]
    g_flat = g.view(B, -1)
    q = torch.quantile(g_flat, 0.95, dim=1, keepdim=True) + eps
    m = torch.clamp(g_flat / q, 0.0, 1.0).view_as(g)
    return m.pow(power).detach()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[..., 1:] - x[..., :-1]).abs().mean() + (x[..., 1:, :] - x[..., :-1, :]).abs().mean()


# ----------------------------
# 仿真指标（需要 GT）- 仅用于分析
# ----------------------------
def spectral_angle_deg(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    dot = (pred * gt).sum(dim=1, keepdim=True)
    n1 = torch.sqrt((pred * pred).sum(dim=1, keepdim=True) + eps)
    n2 = torch.sqrt((gt * gt).sum(dim=1, keepdim=True) + eps)
    cos = torch.clamp(dot / (n1 * n2 + eps), -1 + 1e-6, 1 - 1e-6)
    return float((torch.acos(cos) * (180.0 / np.pi)).mean().detach().cpu())


def ergas(pred: torch.Tensor, gt: torch.Tensor, ratio: int = 4, eps: float = 1e-12) -> float:
    B, C, H, W = pred.shape
    pred_c = pred.reshape(B, C, -1)
    gt_c = gt.reshape(B, C, -1)
    rmse_c = torch.sqrt(((pred_c - gt_c) ** 2).mean(dim=2) + eps)
    mean_c = gt_c.mean(dim=2).abs() + eps
    val = ((rmse_c / mean_c) ** 2).mean(dim=1)
    out = 100.0 / ratio * torch.sqrt(val + eps)
    return float(out.mean().detach().cpu())


def psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    mse = ((pred - gt) ** 2).mean()
    return float((10.0 * torch.log10(1.0 / (mse + eps))).detach().cpu())


def ssim_single_channel(x: torch.Tensor, y: torch.Tensor, k1=0.01, k2=0.03) -> torch.Tensor:
    c1 = k1 ** 2
    c2 = k2 ** 2
    mu_x = gaussian_blur(x, sigma=1.5, ksize=11)
    mu_y = gaussian_blur(y, sigma=1.5, ksize=11)
    sigma_x = gaussian_blur(x * x, sigma=1.5, ksize=11) - mu_x * mu_x
    sigma_y = gaussian_blur(y * y, sigma=1.5, ksize=11) - mu_y * mu_y
    sigma_xy = gaussian_blur(x * y, sigma=1.5, ksize=11) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (num / (den + 1e-12)).mean()


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    vals = [ssim_single_channel(pred[:, c:c + 1], gt[:, c:c + 1]) for c in range(pred.shape[1])]
    return float(torch.stack(vals).mean().detach().cpu())


# ----------------------------
# 旧版 QNR（仅用于记录日志）
# ----------------------------
def q_index(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B = x.shape[0]
    xf = x.view(B, -1)
    yf = y.view(B, -1)
    mx = xf.mean(dim=1, keepdim=True)
    my = yf.mean(dim=1, keepdim=True)
    vx = (xf - mx).pow(2).mean(dim=1, keepdim=True)
    vy = (yf - my).pow(2).mean(dim=1, keepdim=True)
    cxy = ((xf - mx) * (yf - my)).mean(dim=1, keepdim=True)
    num = 4 * mx * my * cxy
    den = (mx * mx + my * my) * (vx + vy) + eps
    return (num / den).squeeze(1)


@torch.no_grad()
def legacy_qnr(fused: torch.Tensor, ms_up: torch.Tensor, pan: torch.Tensor,
               alpha: float = 1.0, beta: float = 1.0) -> Dict[str, float]:
    fused_lp = gaussian_blur(fused, sigma=1.6, ksize=11)
    ms_lp = gaussian_blur(ms_up, sigma=1.6, ksize=11)
    B, C, H, W = fused.shape

    dlam = 0.0
    cnt = 0
    for i in range(C):
        for j in range(i + 1, C):
            q_f = q_index(fused_lp[:, i:i+1], fused_lp[:, j:j+1]).mean()
            q_m = q_index(ms_lp[:, i:i+1], ms_lp[:, j:j+1]).mean()
            dlam = dlam + torch.abs(q_f - q_m)
            cnt += 1
    D_lambda = (dlam / max(1, cnt)).item()

    pan_hp = high_pass(pan, sigma=2.0)
    Ds = 0.0
    for i in range(C):
        q_f = q_index(high_pass(fused[:, i:i+1], sigma=2.0), pan_hp).mean()
        q_m = q_index(high_pass(ms_up[:, i:i+1], sigma=2.0), pan_hp).mean()
        Ds = Ds + torch.abs(q_f - q_m)
    D_s = (Ds / C).item()

    QNR = ((1.0 - D_lambda) ** alpha) * ((1.0 - D_s) ** beta)
    return {"D_lambda": float(D_lambda), "D_s": float(D_s), "QNR": float(QNR)}


# ----------------------------
# 数据集（+ 简单几何增强）
# ----------------------------
class PanMsDataset(Dataset):
    def __init__(self, folder: str, use_gt: bool = False, augment: bool = False):
        self.folder = folder
        self.use_gt = use_gt
        self.augment = augment
        self.pan_files = sorted(glob.glob(os.path.join(folder, "*_pan.tif")))
        if len(self.pan_files) == 0:
            raise FileNotFoundError(f"No *_pan.tif found in {folder}")

    def __len__(self):
        return len(self.pan_files)

    def _aug(self, pan: torch.Tensor, ms: torch.Tensor, gt: torch.Tensor = None):
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            pan = torch.rot90(pan, k, dims=(1, 2))
            ms = torch.rot90(ms, k, dims=(1, 2))
            if gt is not None:
                gt = torch.rot90(gt, k, dims=(1, 2))
        if torch.rand(1).item() < 0.5:
            pan = torch.flip(pan, dims=(2,))
            ms = torch.flip(ms, dims=(2,))
            if gt is not None:
                gt = torch.flip(gt, dims=(2,))
        if torch.rand(1).item() < 0.5:
            pan = torch.flip(pan, dims=(1,))
            ms = torch.flip(ms, dims=(1,))
            if gt is not None:
                gt = torch.flip(gt, dims=(1,))
        return pan, ms, gt

    def __getitem__(self, idx):
        pan_path = self.pan_files[idx]
        base = pan_path.replace("_pan.tif", "")
        mul_path = base + "_mul.tif"
        ymul_path = base + "_ymul.tif"

        pan = normalize_01(read_tif_chw_keep_dtype(pan_path)).astype(np.float32)
        ms_up = normalize_01(read_tif_chw_keep_dtype(mul_path)).astype(np.float32)

        pan_t = torch.from_numpy(pan)
        ms_t = torch.from_numpy(ms_up)
        name = os.path.basename(base)

        if self.use_gt:
            gt = normalize_01(read_tif_chw_keep_dtype(ymul_path)).astype(np.float32)
            gt_t = torch.from_numpy(gt)
            if self.augment:
                pan_t, ms_t, gt_t = self._aug(pan_t, ms_t, gt_t)
            return pan_t, ms_t, gt_t, name

        if self.augment:
            pan_t, ms_t, _ = self._aug(pan_t, ms_t, None)
        return pan_t, ms_t, name


# ----------------------------
# 真实测试数据集（仅 PAN + MS_up，无 GT）
# ----------------------------
def _stem_key(stem: str) -> str:
    """

    例如:
      M0-1_pan -> M0-1
      M11-3_mul -> M11-3
      abc_xyz_pan -> abc_xyz
    """
    st = stem
    for suf in ["_pan", "_PAN", "_mul", "_MUL", "_ms", "_MS", "_msup", "_MSUP", "_ms_up", "_MS_up", "_ymul", "_YMUL"]:
        if st.endswith(suf):
            st = st[: -len(suf)]
            break
    if "_" in st:
        # 如果末尾仍带有 *_pan_cut 等尾缀，只移除最后一个 token
        # 但保留连字符以及前面的下划线部分。
        parts = st.split("_")
        if parts[-1] in ("pan", "mul", "ms", "msup", "ymul"):
            st = "_".join(parts[:-1])
    return st


class RealPanMsDataset(Dataset):
    """真实分辨率测试数据集（无 GT）。

    需要两个目录：
      - ms_dir：上采样后的多光谱影像（与 PAN 具有相同的 H/W），例如 MS_up_800
      - pan_dir：全色影像，例如 PAN_cut_800
    配对方式：根据文件名 key 进行匹配。
    """

    def __init__(self, ms_dir: str, pan_dir: str):
        self.ms_dir = ms_dir
        self.pan_dir = pan_dir
        self.ms_files = sorted(glob.glob(os.path.join(ms_dir, "*.tif")))
        self.pan_files = sorted(glob.glob(os.path.join(pan_dir, "*.tif")))
        if not self.ms_files:
            raise FileNotFoundError(f"No tif found in ms_dir: {ms_dir}")
        if not self.pan_files:
            raise FileNotFoundError(f"No tif found in pan_dir: {pan_dir}")

        ms_map = {}
        for p in self.ms_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            ms_map[_stem_key(stem)] = p

        pairs = []
        for p in self.pan_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            k = _stem_key(stem)
            if k in ms_map:
                pairs.append((k, ms_map[k], p))

        if not pairs:
            raise FileNotFoundError(
                "No paired files found. Make sure MS/PAN filenames share the same prefix, e.g. M0-1_mul.tif with M0-1_pan.tif."
            )

        # 按 key 做确定性排序
        pairs = sorted(pairs, key=lambda x: x[0])
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        key, ms_path, pan_path = self.pairs[idx]
        ms = normalize_01(read_tif_chw_keep_dtype(ms_path)).astype(np.float32)
        pan = normalize_01(read_tif_chw_keep_dtype(pan_path)).astype(np.float32)
        ms_t = torch.from_numpy(ms)
        pan_t = torch.from_numpy(pan)
        return pan_t, ms_t, key, ms_path, pan_path



def infer_ms_ch_from_folder(folder: str) -> int:
    mul_files = sorted(glob.glob(os.path.join(folder, "*_mul.tif")))
    if not mul_files:
        raise FileNotFoundError(f"No *_mul.tif found in {folder}")
    ms = read_tif_chw_keep_dtype(mul_files[0])
    return int(_to_chw(ms).shape[0])


# ----------------------------
# PAN 强度映射权重（自适应）
# ----------------------------
@torch.no_grad()
def estimate_pan_weights(ms: torch.Tensor, pan: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    B, C, H, W = ms.shape
    X = ms.view(B, C, -1).permute(0, 2, 1)  # 形状：B,N,C
    y = pan.view(B, 1, -1).permute(0, 2, 1)  # 形状：B,N,1
    XtX = X.transpose(1, 2) @ X
    Xty = X.transpose(1, 2) @ y
    I = torch.eye(C, device=ms.device, dtype=ms.dtype).unsqueeze(0).repeat(B, 1, 1)
    w = torch.linalg.solve(XtX + lam * I, Xty).squeeze(-1)  # 形状：B,C
    return w


# ----------------------------
# 模型：用于逐波段残差的小型 UNet
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.GroupNorm(num_groups=max(1, c_out // 8), num_channels=c_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, 1, 1),
            nn.GroupNorm(num_groups=max(1, c_out // 8), num_channels=c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetResidual(nn.Module):
    def __init__(self, ms_ch: int, base: int = 32, r_max: float = 0.25):
        super().__init__()
        self.r_max = r_max
        in_ch = ms_ch + 1

        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.pool = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 4, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.dec1 = ConvBlock(base * 2, base)

        self.head = nn.Conv2d(base, ms_ch, 3, 1, 1)

    def forward(self, ms_up: torch.Tensor, pan: torch.Tensor, r_scale: float = 1.0):
        x = torch.cat([ms_up, pan], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(e3)

        d2 = self.up2(m)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        r = torch.tanh(self.head(d1)) * self.r_max * r_scale
        r = r - r.mean(dim=(2, 3), keepdim=True)
        return r


# ----------------------------
# 损失函数
# ----------------------------
def masked_ncc_loss(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    B = a.shape[0]
    a_f = a.view(B, -1)
    b_f = b.view(B, -1)
    m_f = m.view(B, -1)
    msum = m_f.sum(dim=1, keepdim=True) + eps
    ma = (a_f * m_f).sum(dim=1, keepdim=True) / msum
    mb = (b_f * m_f).sum(dim=1, keepdim=True) / msum
    a0 = (a_f - ma) * m_f
    b0 = (b_f - mb) * m_f
    num = (a0 * b0).sum(dim=1)
    den = torch.sqrt((a0 * a0).sum(dim=1) * (b0 * b0).sum(dim=1) + eps)
    corr = torch.clamp(num / (den + eps), -1.0, 1.0)
    return (1.0 - corr).mean()


def grad_dir_loss(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    ax, ay = sobel_gx_gy(a)
    bx, by = sobel_gx_gy(b)
    an = torch.sqrt(ax * ax + ay * ay + eps)
    bn = torch.sqrt(bx * bx + by * by + eps)
    cos = (ax/(an+eps))*(bx/(bn+eps)) + (ay/(an+eps))*(by/(bn+eps))
    return (m * (1.0 - cos)).mean()


@dataclass
class LossWeights:
    w_pan_ncc: float = 1.6
    w_pan_dir: float = 0.6
    w_lp: float = 0.15
    w_res_l1: float = 0.08
    w_tv: float = 3e-4


def fuse_forward(ms_up: torch.Tensor, pan: torch.Tensor, model: UNetResidual, scale: int, t: float,
                 edge_gate: int = 1, edge_pow: float = 1.2):
    # 残差强度日程：先热身，再缓慢增长
    r_scale = 0.25 + 0.75 * t
    r_hr = model(ms_up, pan, r_scale=r_scale)

    B, C, H, W = r_hr.shape
    # 投影到下采样算子的零空间：
    # r_ns = r - Up(Down(r))（零空间残差）
    r_lr = degrade_to_lr(r_hr, scale=scale)
    r_lr_up = upsample_to_hr(r_lr, (H, W))
    r_ns = r_hr - r_lr_up

    # 去除 r_ns 中的超低频漂移（有助于颜色稳定）
    r_ns = r_ns - gaussian_blur(r_ns, sigma=2.0, ksize=11)

    gate = None
    if edge_gate:
        # 由 PAN 的高频结构计算边缘掩码
        gate = edge_mask_from_pan(pan, power=edge_pow)  # 形状：B,1,H,W
        r_ns = r_ns * gate  # 主要在边缘处注入细节

    fused = torch.clamp(ms_up + r_ns, 0.0, 1.0)

    # 自适应 PAN 强度映射权重（逐图回归）
    w = estimate_pan_weights(ms_up.detach(), pan.detach())
    I_fused = (fused * w.view(-1, C, 1, 1)).sum(dim=1, keepdim=True)

    cache = {"r_scale": r_scale, "r_ns": r_ns, "I_fused": I_fused}
    if gate is not None:
        cache["gate"] = gate
    return fused, cache


def selfsup_loss(pan: torch.Tensor, ms_up: torch.Tensor, fused: torch.Tensor, cache: Dict[str, torch.Tensor],
                 lw: LossWeights, w_bg: float = 0.12):
    I_fused = cache["I_fused"]
    r_ns = cache["r_ns"]

    pan_hp = high_pass(pan, sigma=2.0)
    I_hp = high_pass(I_fused, sigma=2.0)
    M = edge_mask_from_pan(pan, power=1.0)

    # 空间细节对齐（侧重边缘）
    L_pan = masked_ncc_loss(I_hp, pan_hp, M)
    L_dir = grad_dir_loss(I_hp, pan_hp, M)

    # 低频光谱锚定（保持颜色稳定）
    fused_lp = gaussian_blur(fused, sigma=2.0, ksize=11)
    ms_lp = gaussian_blur(ms_up, sigma=2.0, ksize=11)
    L_lp = F.mse_loss(fused_lp, ms_lp)

    # 对注入的细节做正则化
    L_res = r_ns.abs().mean()
    L_tv = tv_loss(r_ns)

    # 新增：抑制非边缘区域的细节注入
    if "gate" in cache:
        gate = cache["gate"]  # 形状：B,1,H,W
        L_bg = ((1.0 - gate) * r_ns.abs()).mean()
    else:
        L_bg = torch.tensor(0.0, device=pan.device, dtype=pan.dtype)

    loss = (lw.w_pan_ncc * L_pan +
            lw.w_pan_dir * L_dir +
            lw.w_lp * L_lp +
            lw.w_res_l1 * L_res +
            lw.w_tv * L_tv +
            w_bg * L_bg)

    logs = {
        "loss": float(loss.detach().cpu()),
        "pan_ncc": float(L_pan.detach().cpu()),
        "pan_dir": float(L_dir.detach().cpu()),
        "lp_mse": float(L_lp.detach().cpu()),
        "res": float(L_res.detach().cpu()),
        "bg": float(L_bg.detach().cpu()),
    }
    return loss, logs


@torch.no_grad()
def selfscore(pan: torch.Tensor, ms_up: torch.Tensor, fused: torch.Tensor, cache: Dict[str, torch.Tensor],
              c_res: float = 0.6, c_lp: float = 300.0) -> Dict[str, float]:
    I_fused = cache["I_fused"]
    r_ns = cache["r_ns"]

    pan_hp = high_pass(pan, sigma=2.0)
    I_hp = high_pass(I_fused, sigma=2.0)
    M = edge_mask_from_pan(pan, power=1.0)
    corr = float((1.0 - masked_ncc_loss(I_hp, pan_hp, M)).detach().cpu())

    fused_lp = gaussian_blur(fused, sigma=2.0, ksize=11)
    ms_lp = gaussian_blur(ms_up, sigma=2.0, ksize=11)
    lp_mse = float(F.mse_loss(fused_lp, ms_lp).detach().cpu())

    res = float(r_ns.abs().mean().detach().cpu())
    score = corr - c_res * res - c_lp * lp_mse
    return {"score": float(score), "corr": corr, "lp_mse": lp_mse, "res": res}


# ----------------------------
# 辅助函数：保存图像 + CSV 曲线绘制
# ----------------------------
def save_tif_chw(path: str, arr_chw: np.ndarray):
    ensure_dir(os.path.dirname(path))
    tiff.imwrite(path, np.transpose(arr_chw, (1, 2, 0)).astype(np.float32))


def write_history_csv(path: str, rows: List[Dict[str, float]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join([f"{r[k]:.8f}" for k in keys]) + "\n")


def plot_history(history_csv: str, out_dir: str):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import csv
    import matplotlib.pyplot as plt

    with open(history_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    epochs = [int(float(r["epoch"])) for r in rows]

    def series(key: str):
        return [float(r[key]) for r in rows]

    plots = [
        ("train_loss", "Train loss", "train_loss.png"),
        ("train_pan_ncc", "Train pan_ncc (↓)", "train_pan_ncc.png"),
        ("train_lp_mse", "Train lp_mse (↓)", "train_lp_mse.png"),
        ("train_res", "Train |res| (mean)", "train_res.png"),
        ("val_score", "Val SelfScore (↑)", "val_score.png"),
        ("val_corr", "Val corr (↑)", "val_corr.png"),
        ("val_lp_mse", "Val lp_mse (↓)", "val_lp_mse.png"),
        ("val_res", "Val |res| (mean)", "val_res.png"),
        ("val_QNR", "Val legacy QNR (log)", "val_qnr.png"),
        ("val_PSNR", "Val PSNR (GT, sim)", "val_psnr.png"),
        ("val_SSIM", "Val SSIM (GT, sim)", "val_ssim.png"),
        ("val_ERGAS", "Val ERGAS (GT, sim)", "val_ergas.png"),
        ("val_SAM_deg", "Val SAM (GT, sim)", "val_sam.png"),
    ]
    for k, title, fname in plots:
        if k not in rows[0]:
            continue
        plt.figure()
        plt.plot(epochs, series(k))
        plt.xlabel("epoch")
        plt.title(title)
        plt.savefig(os.path.join(out_dir, fname), dpi=160, bbox_inches="tight")
        plt.close()


# ----------------------------
# 基线 / 评估 / 训练 / 测试
# ----------------------------
@torch.no_grad()
def baseline_sim(args):
    device = args.device
    test_dir = pick_test_dir(args.data_root)
    ds = PanMsDataset(test_dir, use_gt=True, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    ensure_dir(args.out_dir)

    sams, erg, psn, ssi = [], [], [], []
    for _pan, ms_up, gt, _name in tqdm(dl, desc="BaselineSim (MS_up)", ncols=110):
        ms_up = ms_up.to(device)
        gt = gt.to(device)
        sams.append(spectral_angle_deg(ms_up, gt))
        erg.append(ergas(ms_up, gt, ratio=args.scale))
        psn.append(psnr(ms_up, gt))
        ssi.append(ssim(ms_up, gt))

    print(f"\n[Baseline Summary: MS_up as output] SAM(deg)={float(np.mean(sams)):.4f} ERGAS={float(np.mean(erg)):.4f} PSNR={float(np.mean(psn)):.4f} SSIM={float(np.mean(ssi)):.4f}")


@torch.no_grad()
def eval_epoch(model: UNetResidual, dl_gt: DataLoader, dl_nogt: DataLoader, device: str,
               scale: int, t: float, score_c_res: float, score_c_lp: float, score_c_ds: float,
               edge_gate: int, edge_pow: float) -> Dict[str, float]:
    model.eval()

    scores, corrs, lps, ress = [], [], [], []
    qnrs, dls, dss = [], [], []

    for pan, ms_up, _name in dl_nogt:
        pan = pan.to(device)
        ms_up = ms_up.to(device)
        fused, cache = fuse_forward(ms_up, pan, model, scale=scale, t=t, edge_gate=edge_gate, edge_pow=edge_pow)

        sc = selfscore(pan, ms_up, fused, cache, c_res=score_c_res, c_lp=score_c_lp)
        q = legacy_qnr(fused, ms_up, pan)

        score = sc["corr"] - score_c_res * sc["res"] - score_c_lp * sc["lp_mse"] - score_c_ds * q["D_s"]

        scores.append(float(score))
        corrs.append(sc["corr"])
        lps.append(sc["lp_mse"])
        ress.append(sc["res"])

        qnrs.append(q["QNR"])
        dls.append(q["D_lambda"])
        dss.append(q["D_s"])

    out = {
        "score": float(np.mean(scores)),
        "corr": float(np.mean(corrs)),
        "lp_mse": float(np.mean(lps)),
        "res": float(np.mean(ress)),
        "QNR": float(np.mean(qnrs)),
        "D_lambda": float(np.mean(dls)),
        "D_s": float(np.mean(dss)),
    }

    sams, ergs, psnrs, ssims = [], [], [], []
    for pan, ms_up, gt, _name in dl_gt:
        pan = pan.to(device)
        ms_up = ms_up.to(device)
        gt = gt.to(device)
        fused, _ = fuse_forward(ms_up, pan, model, scale=scale, t=t, edge_gate=edge_gate, edge_pow=edge_pow)
        sams.append(spectral_angle_deg(fused, gt))
        ergs.append(ergas(fused, gt, ratio=scale))
        psnrs.append(psnr(fused, gt))
        ssims.append(ssim(fused, gt))

    out.update({
        "SAM_deg": float(np.mean(sams)),
        "ERGAS": float(np.mean(ergs)),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
    })
    return out


def train(args):
    set_seed(args.seed)
    device = args.device

    train_dir = pick_train_dir(args.data_root)
    test_dir = pick_test_dir(args.data_root)

    if args.ms_ch <= 0:
        args.ms_ch = infer_ms_ch_from_folder(train_dir)
        print(f"[Auto] ms_ch inferred as {args.ms_ch}")

    ensure_dir(args.out_dir)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    ds_train = PanMsDataset(train_dir, use_gt=False, augment=True)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True)

    ds_val_nogt = PanMsDataset(test_dir, use_gt=False, augment=False)
    dl_val_nogt = DataLoader(ds_val_nogt, batch_size=args.val_batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    ds_val_gt = PanMsDataset(test_dir, use_gt=True, augment=False)
    dl_val_gt = DataLoader(ds_val_gt, batch_size=args.val_batch_size, shuffle=False,
                           num_workers=args.num_workers, drop_last=False)

    model = UNetResidual(ms_ch=args.ms_ch, base=args.base, r_max=args.r_max).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.2)

    lw = LossWeights(
        w_pan_ncc=args.w_pan_ncc,
        w_pan_dir=args.w_pan_dir,
        w_lp=args.w_lp,
        w_res_l1=args.w_res_l1,
        w_tv=args.w_tv
    )

    wandb = None
    run = None
    if args.wandb:
        wandb = try_import_wandb()
        if wandb is None:
            print("[WARN] --wandb set but wandb not installed. pip install wandb")
        else:
            run = wandb.init(project=args.wandb_project, name=args.wandb_name or None,
                             config=vars(args) | asdict(lw), dir=args.out_dir, reinit=True)

    history: List[Dict[str, float]] = []
    global_step = 0

    best_score = -1e18
    best_epoch = 0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t = (epoch - 1) / max(1, args.epochs - 1)

        sum_loss = sum_pan = sum_lp = sum_res = 0.0
        n_batches = 0

        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", ncols=140)
        for pan, ms_up, _name in pbar:
            pan = pan.to(device)
            ms_up = ms_up.to(device)

            fused, cache = fuse_forward(ms_up, pan, model, scale=args.scale, t=t, edge_gate=args.edge_gate, edge_pow=args.edge_pow)
            loss, logs = selfsup_loss(pan, ms_up, fused, cache, lw, w_bg=args.w_bg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            sum_loss += logs["loss"]
            sum_pan += logs["pan_ncc"]
            sum_lp += logs["lp_mse"]
            sum_res += logs["res"]
            n_batches += 1

            lr_now = opt.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{logs['loss']:.4f}", "pan": f"{logs['pan_ncc']:.4f}",
                              "lp": f"{logs['lp_mse']:.6f}", "res": f"{logs['res']:.4f}",
                              "lr": f"{lr_now:.6g}", "t": f"{t:.2f}"})

            if run is not None and (global_step % args.wandb_log_every == 0):
                run.log({
                    "train/loss": logs["loss"],
                    "train/pan_ncc": logs["pan_ncc"],
                    "train/pan_dir": logs["pan_dir"],
                    "train/lp_mse": logs["lp_mse"],
                    "train/res": logs["res"],
                    "train/lr": lr_now,
                    "train/t": t,
                    "train/r_scale": float(cache["r_scale"]),
                }, step=global_step)

            global_step += 1

        sched.step()

        train_loss = sum_loss / max(1, n_batches)
        train_pan = sum_pan / max(1, n_batches)
        train_lp = sum_lp / max(1, n_batches)
        train_res = sum_res / max(1, n_batches)

        torch.save({
            "epoch": epoch, "ms_ch": args.ms_ch, "scale": args.scale,
            "base": args.base, "r_max": args.r_max,
            "model": model.state_dict()
        }, os.path.join(ckpt_dir, "checkpoint_last.pt"))

        met = eval_epoch(model, dl_val_gt, dl_val_nogt, device, args.scale, t,
                         score_c_res=args.score_c_res, score_c_lp=args.score_c_lp, score_c_ds=args.score_c_ds,
                         edge_gate=args.edge_gate, edge_pow=args.edge_pow)

        print(f"\n[Val @ epoch {epoch}] score={met['score']:.4f} corr={met['corr']:.4f} "
              f"lp_mse={met['lp_mse']:.6f} res={met['res']:.4f} | "
              f"QNR(log)={met['QNR']:.4f} (Dl={met['D_lambda']:.4f}, Ds={met['D_s']:.4f}) | "
              f"SAM={met['SAM_deg']:.4f} ERGAS={met['ERGAS']:.4f} PSNR={met['PSNR']:.4f} SSIM={met['SSIM']:.4f}")

        if met["score"] > best_score:
            best_score = met["score"]
            best_epoch = epoch
            patience = 0
            torch.save({
                "epoch": epoch, "ms_ch": args.ms_ch, "scale": args.scale,
                "base": args.base, "r_max": args.r_max,
                "model": model.state_dict(), "best_score": best_score
            }, os.path.join(ckpt_dir, "checkpoint_best_score.pt"))
            print(f"[Save] best_score -> checkpoint_best_score.pt (score={best_score:.4f} @ epoch {best_epoch})")
        else:
            patience += 1

        history.append({
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_pan_ncc": float(train_pan),
            "train_lp_mse": float(train_lp),
            "train_res": float(train_res),
            "val_score": float(met["score"]),
            "val_corr": float(met["corr"]),
            "val_lp_mse": float(met["lp_mse"]),
            "val_res": float(met["res"]),
            "val_QNR": float(met["QNR"]),
            "val_D_lambda": float(met["D_lambda"]),
            "val_D_s": float(met["D_s"]),
            "val_SAM_deg": float(met["SAM_deg"]),
            "val_ERGAS": float(met["ERGAS"]),
            "val_PSNR": float(met["PSNR"]),
            "val_SSIM": float(met["SSIM"]),
        })
        hist_csv = os.path.join(args.out_dir, "history.csv")
        write_history_csv(hist_csv, history)
        if epoch % args.plot_every == 0 or epoch == args.epochs:
            try:
                plot_history(hist_csv, args.out_dir)
            except Exception as e:
                print(f"[WARN] plot failed: {e}")

        if run is not None:
            run.log({
                "epoch": epoch,
                "val/score": met["score"],
                "val/corr": met["corr"],
                "val/lp_mse": met["lp_mse"],
                "val/res": met["res"],
                "val/QNR": met["QNR"],
                "val/D_lambda": met["D_lambda"],
                "val/D_s": met["D_s"],
                "val/SAM_deg": met["SAM_deg"],
                "val/ERGAS": met["ERGAS"],
                "val/PSNR": met["PSNR"],
                "val/SSIM": met["SSIM"],
                "train_epoch/loss": train_loss,
            }, step=global_step)

        if args.early_stop and patience >= args.patience:
            print(f"[EarlyStop] No score improvement for {args.patience} epochs. Best score={best_score:.4f} @ epoch {best_epoch}.")
            break

    if run is not None:
        run.finish()

    print("Training done.")
    print(f"[Saved] history.csv -> {os.path.join(args.out_dir, 'history.csv')}")
    print(f"[Best] score={best_score:.4f} @ epoch {best_epoch} -> {os.path.join(ckpt_dir, 'checkpoint_best_score.pt')}")


@torch.no_grad()
def test_sim(args):
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    ms_ch = ckpt["ms_ch"]

    model = UNetResidual(ms_ch=ms_ch, base=ckpt.get("base", 32), r_max=ckpt.get("r_max", 0.25)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    test_dir = pick_test_dir(args.data_root)
    ds = PanMsDataset(test_dir, use_gt=True, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    ensure_dir(args.out_dir)
    save_n = max(0, int(args.save_n))

    sams, ergs, psnrs, ssims = [], [], [], []
    saved = 0
    for pan, ms_up, gt, name in tqdm(dl, desc="SimTest", ncols=110):
        pan = pan.to(device)
        ms_up = ms_up.to(device)
        gt = gt.to(device)

        fused, _ = fuse_forward(ms_up, pan, model, scale=args.scale, t=float(args.test_t), edge_gate=int(args.edge_gate), edge_pow=float(args.edge_pow))

        sams.append(spectral_angle_deg(fused, gt))
        ergs.append(ergas(fused, gt, ratio=args.scale))
        psnrs.append(psnr(fused, gt))
        ssims.append(ssim(fused, gt))

        if save_n > 0 and saved < save_n:
            fused_np = fused.detach().cpu().numpy()
            ms_np = ms_up.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()
            pan_np = pan.detach().cpu().numpy()
            B = fused_np.shape[0]
            for i in range(B):
                if saved >= save_n:
                    break
                stem = name[i]
                save_tif_chw(os.path.join(args.out_dir, f"{stem}_fused.tif"), fused_np[i])
                save_tif_chw(os.path.join(args.out_dir, f"{stem}_msup.tif"), ms_np[i])
                save_tif_chw(os.path.join(args.out_dir, f"{stem}_gt.tif"), gt_np[i])
                save_tif_chw(os.path.join(args.out_dir, f"{stem}_pan.tif"), pan_np[i])
                saved += 1

    print(f"\n[SimTest Summary] SAM(deg)={float(np.mean(sams)):.4f} ERGAS={float(np.mean(ergs)):.4f} "
          f"PSNR={float(np.mean(psnrs)):.4f} SSIM={float(np.mean(ssims)):.4f}")
    if save_n > 0:
        print(f"[Saved] fused / msup / gt / pan tifs -> {args.out_dir}")


@torch.no_grad()
def test_real(args):
    """对真实分辨率的 PAN/MS_up 图像对进行融合（无 GT）。

    保存内容：
      - 每张图的 fused tif
      - （可选）输入（ms_up、pan）tif
      - 含自监督评分的汇总 csv（score、QNR、D_lambda、D_s）
    """
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    ms_ch = ckpt["ms_ch"]

    model = UNetResidual(ms_ch=ms_ch, base=ckpt.get("base", 32), r_max=ckpt.get("r_max", 0.25)).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    ds = RealPanMsDataset(args.ms_dir, args.pan_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    ensure_dir(args.out_dir)
    out_fused = os.path.join(args.out_dir, "fused_tif")
    out_inp = os.path.join(args.out_dir, "inputs_tif")
    out_png = os.path.join(args.out_dir, "quicklook_png")
    ensure_dir(out_fused)
    if int(args.save_inputs):
        ensure_dir(out_inp)
    if bool(args.save_png):
        ensure_dir(out_png)

    # 解析 RGB 波段索引
    rgb = [int(x) for x in str(args.rgb).split(",")]
    if len(rgb) != 3:
        rgb = [2, 1, 0]

    def _to_uint8_rgb(x_chw: np.ndarray) -> np.ndarray:
        # x_chw：C,H,W，取值范围 [0,1]
        c, h, w = x_chw.shape
        idx = [min(max(i, 0), c - 1) for i in rgb]
        img = np.stack([x_chw[idx[0]], x_chw[idx[1]], x_chw[idx[2]]], axis=-1)
        # 对比度拉伸
        lo = np.percentile(img, 1.0)
        hi = np.percentile(img, 99.0)
        if hi - lo < 1e-6:
            img = np.clip(img, 0.0, 1.0)
        else:
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return (img * 255.0 + 0.5).astype(np.uint8)

    def _to_uint8_gray(x_chw: np.ndarray) -> np.ndarray:
        img = x_chw[0]
        lo = np.percentile(img, 1.0)
        hi = np.percentile(img, 99.0)
        if hi - lo < 1e-6:
            img = np.clip(img, 0.0, 1.0)
        else:
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return (img * 255.0 + 0.5).astype(np.uint8)

    # 可选：使用自监督指标自动选择 t
    t_use = float(args.test_t)
    if bool(args.auto_t):
        t_list = [float(x) for x in str(args.t_list).split(",") if str(x).strip() != ""]
        metric_name = str(args.select_metric).lower()
        best_t, best_val = None, -1e9
        for t in t_list:
            vals = []
            for pan, ms_up, key, ms_path, pan_path in dl:
                pan = pan.to(device)
                ms_up = ms_up.to(device)
                fused, cache = fuse_forward(ms_up, pan, model, scale=args.scale, t=float(t),
                                            edge_gate=int(args.edge_gate), edge_pow=float(args.edge_pow))
                if metric_name == "qnr":
                    qnr = legacy_qnr(fused, ms_up, pan)
                    vals.append(float(qnr["QNR"]))
                else:
                    sc = selfscore(pan, ms_up, fused, cache)
                    vals.append(float(sc["score"]))
            v = float(np.mean(vals)) if vals else -1e9
            if v > best_val:
                best_val, best_t = v, float(t)
        if best_t is not None:
            t_use = best_t
        print(f"[AutoT] select_metric={args.select_metric} best_t={t_use:.3f} best_val={best_val:.6f}")

    # 使用选定的 t 进行最终推理
    rows = []
    for pan, ms_up, key, ms_path, pan_path in tqdm(dl, desc="RealTest", ncols=110):
        pan = pan.to(device)
        ms_up = ms_up.to(device)
        fused, cache = fuse_forward(ms_up, pan, model, scale=args.scale, t=float(t_use),
                                    edge_gate=int(args.edge_gate), edge_pow=float(args.edge_pow))

        qnr = legacy_qnr(fused, ms_up, pan)
        sc = selfscore(pan, ms_up, fused, cache)

        fused_np = fused.detach().cpu().numpy()
        ms_np = ms_up.detach().cpu().numpy()
        pan_np = pan.detach().cpu().numpy()

        B = fused_np.shape[0]
        for i in range(B):
            name = str(key[i])
            save_tif_chw(os.path.join(out_fused, f"{name}_fused.tif"), fused_np[i])
            if int(args.save_inputs):
                save_tif_chw(os.path.join(out_inp, f"{name}_msup.tif"), ms_np[i])
                save_tif_chw(os.path.join(out_inp, f"{name}_pan.tif"), pan_np[i])

            if bool(args.save_png):
                import matplotlib
                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt
                ms_rgb = _to_uint8_rgb(ms_np[i])
                fu_rgb = _to_uint8_rgb(fused_np[i])
                pa_g = _to_uint8_gray(pan_np[i])

                fig = plt.figure(figsize=(10, 3))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.imshow(pa_g, cmap="gray", vmin=0, vmax=255)
                ax1.set_title("PAN")
                ax1.axis("off")

                ax2 = fig.add_subplot(1, 3, 2)
                ax2.imshow(ms_rgb)
                ax2.set_title("MS_up")
                ax2.axis("off")

                ax3 = fig.add_subplot(1, 3, 3)
                ax3.imshow(fu_rgb)
                ax3.set_title(f"Fused (t={t_use:.2f})")
                ax3.axis("off")

                fig.tight_layout()
                fig.savefig(os.path.join(out_png, f"{name}_montage.png"), dpi=160)
                plt.close(fig)

            rows.append({
                "name": name,
                "t": float(t_use),
                "score": float(sc["score"]),
                "corr": float(sc["corr"]),
                "lp_mse": float(sc["lp_mse"]),
                "res": float(sc["res"]),
                "QNR": float(qnr["QNR"]),
                "D_lambda": float(qnr["D_lambda"]),
                "D_s": float(qnr["D_s"]),
            })

    # 写出 CSV 汇总
    csv_path = os.path.join(args.out_dir, "real_test_summary.csv")
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join([str(r[k]) for k in keys]) + "\n")

    # 打印平均指标（无 GT，因此仅自监督指标）
    if rows:
        mean_score = float(np.mean([r["score"] for r in rows]))
        mean_qnr = float(np.mean([r["QNR"] for r in rows]))
        print(f"\n[RealTest Summary] t={t_use:.3f} mean_score={mean_score:.4f} mean_QNR={mean_qnr:.4f}")
        print(f"[Saved] fused tif -> {out_fused}")
        if int(args.save_inputs):
            print(f"[Saved] inputs tif -> {out_inp}")
        if bool(args.save_png):
            print(f"[Saved] quicklook png -> {out_png}")
        print(f"[Saved] csv -> {csv_path}")




# =============================================================================
# real_test 关键参数:
#   --ms_dir / --pan_dir    真实测试集 MS_up 与 PAN 的目录
#   --auto_t               开启 Auto-t：在候选 t 上做无参考打分，自动选最优 t
#   --t_candidates          Auto-t 的候选 t 列表（例如 0.05,0.1,0.15,0.2）
#   --out_dir              输出融合结果目录（保存 fused tif）
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser("Self-supervised PanSharpening v11 (Null-space + Edge-gated injection)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("baseline_sim")
    pb.add_argument("--data_root", type=str, required=True)
    pb.add_argument("--out_dir", type=str, required=True)
    pb.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pb.add_argument("--batch_size", type=int, default=32)
    pb.add_argument("--num_workers", type=int, default=0)
    pb.add_argument("--scale", type=int, default=4)

    pt = sub.add_parser("train")
    pt.add_argument("--data_root", type=str, required=True)
    pt.add_argument("--out_dir", type=str, required=True)
    pt.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pt.add_argument("--seed", type=int, default=1234)
    pt.add_argument("--ms_ch", type=int, default=0)
    pt.add_argument("--scale", type=int, default=4)
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--batch_size", type=int, default=64)
    pt.add_argument("--val_batch_size", type=int, default=32)
    pt.add_argument("--num_workers", type=int, default=0)
    pt.add_argument("--lr", type=float, default=2.5e-4)
    pt.add_argument("--base", type=int, default=32)
    pt.add_argument("--r_max", type=float, default=0.25)
    pt.add_argument("--edge_gate", type=int, default=1, help="1: multiply residual by PAN edge mask")
    pt.add_argument("--edge_pow", type=float, default=1.2, help="edge mask power (>1 tighter)")
    pt.add_argument("--w_bg", type=float, default=0.12, help="penalty for residual in non-edge regions")


    pt.add_argument("--w_pan_ncc", type=float, default=1.6)
    pt.add_argument("--w_pan_dir", type=float, default=0.6)
    pt.add_argument("--w_lp", type=float, default=0.15)
    pt.add_argument("--w_res_l1", type=float, default=0.08)
    pt.add_argument("--w_tv", type=float, default=3e-4)

    pt.add_argument("--score_c_res", type=float, default=0.6)
    pt.add_argument("--score_c_lp", type=float, default=300.0)
    pt.add_argument("--score_c_ds", type=float, default=0.9, help="penalty on D_s in val score")

    pt.add_argument("--plot_every", type=int, default=5)
    pt.add_argument("--early_stop", type=int, default=1)
    pt.add_argument("--patience", type=int, default=60)

    pt.add_argument("--wandb", action="store_true")
    pt.add_argument("--wandb_project", type=str, default="pansharpen-ssl")
    pt.add_argument("--wandb_name", type=str, default="")
    pt.add_argument("--wandb_log_every", type=int, default=25)

    ps = sub.add_parser("test_sim")
    ps.add_argument("--data_root", type=str, required=True)
    ps.add_argument("--ckpt", type=str, required=True)
    ps.add_argument("--out_dir", type=str, required=True)
    ps.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ps.add_argument("--batch_size", type=int, default=32)
    ps.add_argument("--num_workers", type=int, default=0)
    ps.add_argument("--scale", type=int, default=4)
    ps.add_argument("--save_n", type=int, default=0)
    ps.add_argument("--test_t", type=float, default=0.2, help="fusion strength in [0,1]; should match training epoch fraction")
    ps.add_argument("--edge_gate", type=int, default=1, help="use edge gating in test")
    ps.add_argument("--edge_pow", type=float, default=1.2, help="edge mask power for test")

    
    pr = sub.add_parser("test_real")
    pr.add_argument("--ms_dir", type=str, required=True, help="Folder of MS_up tif (same H/W as PAN), e.g. ...\\MS_up_800")
    pr.add_argument("--pan_dir", type=str, required=True, help="Folder of PAN tif, e.g. ...\\PAN_cut_800")
    pr.add_argument("--ckpt", type=str, required=True)
    pr.add_argument("--out_dir", type=str, required=True)
    pr.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pr.add_argument("--batch_size", type=int, default=1)
    pr.add_argument("--num_workers", type=int, default=0)
    pr.add_argument("--scale", type=int, default=4)
    pr.add_argument("--test_t", type=float, default=0.2)
    pr.add_argument("--edge_gate", type=int, default=1)
    pr.add_argument("--edge_pow", type=float, default=1.4)
    pr.add_argument("--auto_t", action="store_true", help="Try multiple t and pick the best by select_metric")
    pr.add_argument("--t_list", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4")
    pr.add_argument("--select_metric", type=str, choices=["score", "QNR"], default="score")
    pr.add_argument("--save_inputs", type=int, default=1, help="1: also save input MS_up and PAN tifs")
    pr.add_argument("--save_png", action="store_true", help="Also save quicklook montages (PAN / MS_up / Fused) as PNG")
    pr.add_argument("--rgb", type=str, default="2,1,0", help="RGB band indices for quicklook, e.g. '4,3,2'")
    return p
def main():
    args = build_parser().parse_args()
    ensure_dir(args.out_dir)
    if args.cmd == "baseline_sim":
        baseline_sim(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "test_sim":
        test_sim(args)
    elif args.cmd == "test_real":
        test_real(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
