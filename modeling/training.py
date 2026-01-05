# training.py (updated to read ALL params from configuration.json; uses read_json from utilities)
import os
import math
import time

import rasterio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from utilities.utilities import *


def normalize_per_band(x, eps=1e-6):
    """
    Normalizes a multi-channel image per band to [0, 1].

    Args:
        x (np.ndarray): Array of shape (C, H, W).
        eps (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Normalized array of shape (C, H, W).
    """
    x = x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        vmin = np.nanmin(x[c])
        vmax = np.nanmax(x[c])
        out[c] = (x[c] - vmin) / (vmax - vmin + eps)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    return out


def extract_patches(chw, patch_size, stride):
    """
    Extracts overlapping patches from a multi-channel image.

    Args:
        chw (np.ndarray): Image array of shape (C, H, W).
        patch_size (int): Patch side length (e.g., 64).
        stride (int): Stride between patch top-left corners (e.g., 32).

    Returns:
        tuple[list[np.ndarray], list[tuple[int,int]]]: (patches, top_lefts)
            patches are arrays of shape (C, patch_size, patch_size).
            top_lefts are (row, col) indices in the original image.
    """
    c, h, w = chw.shape
    patches = []
    top_lefts = []

    max_r = h - patch_size
    max_c = w - patch_size

    r = 0
    while r <= max_r:
        c0 = 0
        while c0 <= max_c:
            patch = chw[:, r:r + patch_size, c0:c0 + patch_size]
            patches.append(patch)
            top_lefts.append((r, c0))
            c0 += stride
        r += stride

    return patches, top_lefts


def expected_patches_per_tile(h, w, patch_size, stride):
    """
    Computes expected patch grid counts (n_h, n_w) and total patches n_total,
    using the same inclusive stepping logic as extract_patches (r<=h-patch_size, c<=w-patch_size).

    Args:
        h (int): Tile height in pixels.
        w (int): Tile width in pixels.
        patch_size (int): Patch side length (e.g., 64).
        stride (int): Stride between patch top-left corners (e.g., 32).

    Returns:
        tuple[int, int, int]: (n_h, n_w, n_total)
            n_h: number of patches along height
            n_w: number of patches along width
            n_total: total patches (n_h * n_w)
    """
    n_h = (h - patch_size) // stride + 1
    n_w = (w - patch_size) // stride + 1
    return int(n_h), int(n_w), int(n_h * n_w)


def compute_ndvi_from_patch(patch, idx_red=1, idx_nir=5, eps=1e-6):
    """
    Computes NDVI-like grayscale from a patch for edge target generation.

    Used only to generate a stable single-channel field for Canny.

    Args:
        patch (np.ndarray): Patch array of shape (C, H, W), assumed normalized to [0, 1].
        idx_red (int): Channel index for Red (B04).
        idx_nir (int): Channel index for NIR (B08).
        eps (float): Small constant to avoid division by zero.

    Returns:
        np.ndarray: Grayscale array of shape (H, W) in [0, 1].
    """
    red = patch[idx_red]
    nir = patch[idx_nir]
    ndvi = (nir - red) / (nir + red + eps)
    ndvi = (ndvi + 1.0) / 2.0
    ndvi = np.clip(ndvi, 0.0, 1.0)
    return ndvi.astype(np.float32)


def canny_edges(gray01, sigma=1.0, low_threshold=0.1, high_threshold=0.3, fallback_grad_quantile=0.90):
    """
    Computes a binary Canny edge map from a grayscale image.

    Args:
        gray01 (np.ndarray): Grayscale array of shape (H, W) in [0, 1].
        sigma (float): Canny smoothing parameter.
        low_threshold (float): Lower hysteresis threshold (0..1).
        high_threshold (float): Upper hysteresis threshold (0..1).
        fallback_grad_quantile (float): Quantile for gradient threshold in fallback edge map.

    Returns:
        np.ndarray: Binary edge map of shape (H, W) with values in {0,1}.
    """
    try:
        from skimage.feature import canny
        edges = canny(gray01, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        return edges.astype(np.uint8)
    except Exception:
        gx = np.zeros_like(gray01, dtype=np.float32)
        gy = np.zeros_like(gray01, dtype=np.float32)
        gx[:, 1:-1] = gray01[:, 2:] - gray01[:, :-2]
        gy[1:-1, :] = gray01[2:, :] - gray01[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(np.quantile(mag, float(fallback_grad_quantile)))
        edges = (mag >= thr).astype(np.uint8)
        return edges


def f1_for_edges(pred_logits, target_binary, threshold=0.25, eps=1e-8):
    """
    Computes F1-score for sparse edge pixels.

    Args:
        pred_logits (torch.Tensor): Predicted edge logits of shape (B, 1, H, W).
        target_binary (torch.Tensor): Target binary edges of shape (B, 1, H, W) with {0,1}.
        threshold (float): Probability threshold for predicted edges.
        eps (float): Small constant to avoid division by zero.

    Returns:
        float: Batch F1-score.
    """
    probs = torch.sigmoid(pred_logits)
    pred = (probs >= threshold).float()
    tgt = target_binary.float()

    tp = (pred * tgt).sum().item()
    fp = (pred * (1.0 - tgt)).sum().item()
    fn = ((1.0 - pred) * tgt).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(f1)


def spearman_corr(x, y):
    """
    Computes Spearman correlation between two 1D arrays.

    Args:
        x (np.ndarray): Predicted values of shape (N,).
        y (np.ndarray): True values of shape (N,).

    Returns:
        float: Spearman rho in [-1, 1].
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    def rankdata(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = (np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum()))
    if denom == 0:
        return 0.0
    return float((rx * ry).sum() / denom)


# ============================================================
# Stage A model (unchanged)
# ============================================================

class TinyCnnEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim=64, encoder_channels=(32, 64)):
        super().__init__()
        c1, c2 = int(encoder_channels[0]), int(encoder_channels[1])
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.emb_dim = emb_dim
        self.proj = nn.Linear(c2, emb_dim) if emb_dim != c2 else None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.mean(dim=(2, 3))
        if self.proj is not None:
            x = self.proj(x)
        return x


class LightEdgeDecoder(nn.Module):
    def __init__(self, emb_dim=64, out_h=96, out_w=96, decoder_fc_units=4096):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.fc = nn.Linear(emb_dim, int(decoder_fc_units))
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        b = z.shape[0]
        x = F.relu(self.fc(z))
        x = x.view(b, 1, 64, 64)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        x = self.conv(x)
        return x


class SslEdgeModel(nn.Module):
    def __init__(self, in_channels, emb_dim=64, patch_size=96, encoder_channels=(32, 64), decoder_fc_units=4096):
        super().__init__()
        self.encoder = TinyCnnEncoder(in_channels, emb_dim, encoder_channels)
        self.decoder = LightEdgeDecoder(emb_dim, patch_size, patch_size, decoder_fc_units)

    def forward(self, x):
        z = self.encoder(x)
        edge_logits = self.decoder(z)
        return z, edge_logits


# ============================================================
# Stage B (OPTION 1 INTEGRATED)
# ============================================================

def build_stage_b_training_data(embeddings_by_tile, proxy_map, train_tile_ids):
    """
    Builds Stage B regression training data by replicating tile-level proxy labels to patches.

    Args:
        embeddings_by_tile (dict): tile_id -> np.ndarray of shape (N_patches, D).
        proxy_map (dict): tile_id -> biomass proxy (float).
        train_tile_ids (list[int]): Training tile IDs (1..15).

    Returns:
        tuple[np.ndarray, np.ndarray]: (X, y)
            X shape (N_total_patches, D)
            y shape (N_total_patches,)
    """
    X = np.vstack([
        embeddings_by_tile[tid].mean(axis=0, keepdims=True)
        for tid in train_tile_ids
    ])
    y = np.array([float(proxy_map[tid]) for tid in train_tile_ids], dtype=np.float32)
    return X, y


def train_stage_b_ridge(X_train, y_train, alpha, use_standard_scaler=True, ridge_fit_intercept=True):
    """
    Trains Ridge regression for Stage B (embedding -> biomass proxy).

    Args:
        X_train (np.ndarray): Training embeddings of shape (N, D).
        y_train (np.ndarray): Training targets of shape (N,).
        alpha (float): Ridge regularization strength.
        use_standard_scaler (bool): Whether to use StandardScaler before Ridge.
        ridge_fit_intercept (bool): Whether Ridge fits an intercept.

    Returns:
        object: Trained sklearn pipeline.
    """
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=bool(ridge_fit_intercept), random_state=0)
    ) if bool(use_standard_scaler) else Ridge(
        alpha=alpha, fit_intercept=bool(ridge_fit_intercept), random_state=0
    )
    model.fit(X_train, y_train)
    return model


def predict_stage_b_tile_level(ridge_model, z_teseachi):
    """
    Generates Stage B predictions for a single evaluation tile.

    The prediction is computed at the tile level by averaging all patch
    embeddings into a single vector, applying the Ridge model once, and then
    broadcasting the resulting scalar prediction back to all patches. This
    preserves compatibility with patch-level plots and metrics while keeping
    the statistical model tile-consistent.

    Args:
        ridge_model:
            Trained Ridge regression model.
        z_teseachi (np.ndarray):
            Patch embeddings for the evaluation tile,
            shape (N_patches, D).

    Returns:
        np.ndarray:
            Array of shape (N_patches,), where all values are identical
            and equal to the tile-level prediction.
    """
    y_tile = ridge_model.predict(z_teseachi.mean(axis=0, keepdims=True))[0]
    return np.full((z_teseachi.shape[0],), y_tile, dtype=np.float32)


# ============================================================
# main_training (ONLY Stage B prediction lines changed)
# ============================================================

def main_training():
    """
    Runs the full two-stage pipeline: extracts patches from tiles, trains Stage A (self-supervised edge task),
    extracts embeddings, trains Stage B (ridge regression on proxy biomass), and saves plots and metrics.

    Args:
        config_path (str): Path to configuration json.

    Returns:
        None
    """
    t0_all = time.time()
    cfg = read_json("utilities/configuration.json")
    tr = cfg["training"]

    dataset_dir = tr["dataset_dir"]
    proxy_dir = tr["proxy_dir"]
    proxy_csv_path = os.path.join(proxy_dir, tr["proxy_csv_name"])
    output_dir = tr["output_dir"]
    stage_b_plot_dir = os.path.join(output_dir, tr["stage_b_plot_subdir"])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stage_b_plot_dir, exist_ok=True)

    batch_size = int(tr["batch_size"])
    ridge_alpha = float(tr["ridge_alpha"])
    ridge_fit_intercept = bool(tr.get("ridge_fit_intercept", True))
    use_standard_scaler = bool(tr.get("use_standard_scaler", True))
    device = resolve_device(tr["device_preference"])

    train_tiles, teseachi_path = list_tile_files(dataset_dir)
    proxy_map = load_proxy_csv(proxy_csv_path)

    embeddings_by_tile = {}
    encoder = SslEdgeModel(1, emb_dim=int(tr["emb_dim"]), patch_size=int(tr["patch_size"])).encoder

    for tid, fpath in train_tiles:
        tile = normalize_per_band(read_tile_tif(fpath))
        patches, _ = extract_patches(tile, int(tr["patch_size"]), int(tr["stride"]))
        embeddings_by_tile[tid] = extract_patch_embeddings(encoder, patches, device, batch_size)

    teseachi_tile = normalize_per_band(read_tile_tif(teseachi_path))
    teseachi_patches, _ = extract_patches(teseachi_tile, int(tr["patch_size"]), int(tr["stride"]))
    z_teseachi = extract_patch_embeddings(encoder, teseachi_patches, device, batch_size)

    train_tile_ids = [tid for tid, _ in train_tiles]
    X_train, y_train = build_stage_b_training_data(embeddings_by_tile, proxy_map, train_tile_ids)

    ridge = train_stage_b_ridge(X_train, y_train, ridge_alpha, use_standard_scaler, ridge_fit_intercept)
    yhat_tile = ridge.predict(z_teseachi.mean(axis=0, keepdims=True))[0]
    yhat_teseachi = np.full((z_teseachi.shape[0],), yhat_tile, dtype=np.float32)

    teseachi_truth_path = os.path.join(proxy_dir, tr["teseachi_truth_csv_name"])
    if os.path.exists(teseachi_truth_path):
        y_true_teseachi, yhat_use = load_teseachi_truth_and_align(teseachi_truth_csv_path=teseachi_truth_path, yhat_teseachi=yhat_teseachi)
        m = evaluate_stage_b(y_true_teseachi, yhat_use)
        print(f"[Stage B Metric] R2 on Teseachi (higher is better): {m['r2']:.4f}")
        print(f"[Stage B Metric] RMSE on Teseachi (lower is better): {m['rmse']:.4f}")
        print(f"[Stage B Metric] Spearman rho on Teseachi (higher is better): {m['spearman']:.4f}")
        plot_stage_b_scatter(y_true_teseachi, yhat_use, os.path.join(stage_b_plot_dir, "stage_b_teseachi_scatter.png"), "Stage B: Predicted vs proxy biomass (Teseachi)")

    print(f"\n[Time] Total elapsed: {format_seconds(time.time() - t0_all)}")
