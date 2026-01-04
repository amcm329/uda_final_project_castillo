import os
import csv
import math 

import rasterio
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def canny_edges(gray01, sigma=1.0, low_threshold=0.1, high_threshold=0.3):
    """
    Computes a binary Canny edge map from a grayscale image.

    Args:
        gray01 (np.ndarray): Grayscale array of shape (H, W) in [0, 1].
        sigma (float): Canny smoothing parameter.
        low_threshold (float): Lower hysteresis threshold (0..1).
        high_threshold (float): Upper hysteresis threshold (0..1).

    Returns:
        np.ndarray: Binary edge map of shape (H, W) with values in {0,1}.
    """
    try:
        from skimage.feature import canny
        edges = canny(gray01, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        return edges.astype(np.uint8)
    except Exception:
        # We fall back to a simple gradient-threshold edge map if skimage is unavailable.
        gx = np.zeros_like(gray01, dtype=np.float32)
        gy = np.zeros_like(gray01, dtype=np.float32)
        gx[:, 1:-1] = gray01[:, 2:] - gray01[:, :-2]
        gy[1:-1, :] = gray01[2:, :] - gray01[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        thr = float(np.quantile(mag, 0.90))
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


def load_teseachi_truth_and_align(teseachi_truth_csv_path, yhat_teseachi):
    """
    Loads Teseachi "measured" biomass targets from CSV and aligns predictions to the same patches.

    Supported CSV formats:

      (A) Full-length truth (one row per extracted patch; no indices):
          biomass
          0.0
          19.34
          ...
          -> Requires len(y_true) == len(yhat_teseachi). Uses row order.

      (B) Subsampled truth (subset with explicit indices):
          patch_index,biomass
          0,19.34
          3,19.34
          ...
          -> Subsets yhat_teseachi by patch_index and aligns to CSV row order.

    IMPORTANT ROBUSTNESS FIX:
      If the CSV patch_index values exceed the available prediction range (0..len(yhat_teseachi)-1),
      we assume the CSV was generated from a different patch grid than the current run.
      In that case, we remap indices from an "old" square grid to the "new" square grid implied by
      len(yhat_teseachi), and we aggregate collisions by averaging the biomass values.

    Args:
        teseachi_truth_csv_path (str): Path to Teseachi truth CSV.
        yhat_teseachi (np.ndarray): Predictions for all extracted Teseachi patches, shape (N,).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            y_true_aligned: True biomass values, shape (M,).
            yhat_aligned:   Predicted biomass values aligned to the same patches, shape (M,).

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If required columns are missing, or lengths cannot be aligned in format (A).
    """
    import csv
    import math

    if not os.path.exists(teseachi_truth_csv_path):
        raise FileNotFoundError(f"Missing Teseachi truth CSV: {teseachi_truth_csv_path}")

    yhat_teseachi = np.asarray(yhat_teseachi, dtype=np.float32).reshape(-1)
    n_pred = int(yhat_teseachi.shape[0])

    indices = []
    y_true = []

    with open(teseachi_truth_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Teseachi truth CSV has no header row.")

        fields = {c.strip() for c in reader.fieldnames if c is not None}
        if "biomass" not in fields:
            raise ValueError(
                "Teseachi truth CSV must contain a 'biomass' column. "
                f"Found columns: {sorted(list(fields))}"
            )

        # Case B: subsampled truth with explicit indices
        if "patch_index" in fields:
            for row in reader:
                idx_str = (row.get("patch_index") or "").strip()
                bio_str = (row.get("biomass") or "").strip()

                if idx_str == "" or bio_str == "":
                    raise ValueError("Found empty 'patch_index' or 'biomass' cell in Teseachi truth CSV.")

                idx = int(float(idx_str))
                indices.append(idx)
                y_true.append(float(bio_str))

            if len(indices) == 0:
                raise ValueError("CSV contains 'patch_index' but no data rows were read.")

            indices_np = np.asarray(indices, dtype=np.int64)
            y_true_np = np.asarray(y_true, dtype=np.float32)

            # If all indices are valid, do the strict alignment.
            if indices_np.min() >= 0 and indices_np.max() < n_pred:
                yhat_aligned = yhat_teseachi[indices_np]
                return y_true_np, yhat_aligned

            # Otherwise, auto-remap old indices -> new indices (square-grid assumption).
            old_max = int(indices_np.max())
            old_n = int(math.ceil(math.sqrt(old_max + 1)))  # old grid width/height (assumed square)

            new_n = int(round(math.sqrt(n_pred)))           # new grid width/height (assumed square)
            if new_n * new_n != n_pred:
                # Fallback: if not a perfect square, clip indices instead of remapping.
                valid_mask = (indices_np >= 0) & (indices_np < n_pred)
                indices_np = indices_np[valid_mask]
                y_true_np = y_true_np[valid_mask]
                if len(indices_np) == 0:
                    raise ValueError(
                        "After clipping invalid patch_index values, no rows remain aligned to current predictions.\n"
                        f"CSV patch_index range was [0, {old_max}], but prediction length is {n_pred}."
                    )
                yhat_aligned = yhat_teseachi[indices_np]
                return y_true_np, yhat_aligned

            # Remap each old (r,c) in old_n x old_n to new (r',c') in new_n x new_n using scaled rounding.
            mapped = []
            for idx in indices_np:
                if idx < 0:
                    continue
                r = int(idx // old_n)
                c = int(idx % old_n)
                r2 = int(round(r * (new_n - 1) / max(old_n - 1, 1)))
                c2 = int(round(c * (new_n - 1) / max(old_n - 1, 1)))
                r2 = int(np.clip(r2, 0, new_n - 1))
                c2 = int(np.clip(c2, 0, new_n - 1))
                mapped.append(r2 * new_n + c2)

            mapped = np.asarray(mapped, dtype=np.int64)
            if mapped.size == 0:
                raise ValueError(
                    "Index remapping produced zero aligned rows. "
                    f"CSV patch_index max={old_max}, prediction length={n_pred}."
                )

            # Aggregate collisions by mean biomass per mapped index.
            agg_sum = {}
            agg_cnt = {}
            for mi, bi in zip(mapped.tolist(), y_true_np.tolist()):
                agg_sum[mi] = agg_sum.get(mi, 0.0) + float(bi)
                agg_cnt[mi] = agg_cnt.get(mi, 0) + 1

            mapped_unique = np.array(sorted(agg_sum.keys()), dtype=np.int64)
            y_true_agg = np.array([agg_sum[k] / agg_cnt[k] for k in mapped_unique.tolist()], dtype=np.float32)
            yhat_agg = yhat_teseachi[mapped_unique]

            return y_true_agg, yhat_agg

        # Case A: full-length truth without indices; must match prediction length
        for row in reader:
            bio_str = (row.get("biomass") or "").strip()
            if bio_str == "":
                raise ValueError("Found empty 'biomass' cell in Teseachi truth CSV.")
            y_true.append(float(bio_str))

    y_true_aligned = np.asarray(y_true, dtype=np.float32)
    if len(y_true_aligned) != n_pred:
        raise ValueError(
            "Teseachi truth CSV has no 'patch_index' column, so it must have one row per extracted patch.\n"
            f"len(y_true)={len(y_true_aligned)} from CSV, len(yhat)={n_pred} from patch extraction.\n"
            "Fix by either (1) regenerating the CSV to full length, or (2) adding 'patch_index' and aligning via indices."
        )

    return y_true_aligned, yhat_teseachi


# ============================================================
# We define Stage A model (Encoder + Decoder)
# ============================================================

class TinyCnnEncoder(nn.Module):
    def __init__(self, in_channels, emb_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.emb_dim = emb_dim

        # We keep the embedding size fixed at 64 via channel count + GAP.
        base_dim = 64
        if emb_dim != base_dim:
            self.proj = nn.Linear(base_dim, emb_dim)
        else:
            self.proj = None

    def forward(self, x):
        # We compute feature maps.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # We apply global average pooling.
        x = x.mean(dim=(2, 3))
        # We optionally project to emb_dim.
        if self.proj is not None:
            x = self.proj(x)
        return x


class LightEdgeDecoder(nn.Module):
    def __init__(self, emb_dim=64, out_h=96, out_w=96):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w

        # We follow the report pattern: Linear(emb_dim->4096), reshape to 1x64x64, upsample, conv.
        self.fc = nn.Linear(emb_dim, 4096)
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        b = z.shape[0]
        x = F.relu(self.fc(z))
        x = x.view(b, 1, 64, 64)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        x = self.conv(x)
        return x


class SslEdgeModel(nn.Module):
    def __init__(self, in_channels, emb_dim=64, patch_size=96):
        super().__init__()
        self.encoder = TinyCnnEncoder(in_channels=in_channels, emb_dim=emb_dim)
        self.decoder = LightEdgeDecoder(emb_dim=emb_dim, out_h=patch_size, out_w=patch_size)

    def forward(self, x):
        z = self.encoder(x)
        edge_logits = self.decoder(z)
        return z, edge_logits


# ============================================================
# We define datasets/loaders (tiles -> patches)
# ============================================================

class PatchSslDataset(Dataset):
    def __init__(self, patches_chw, canny_sigma, canny_low, canny_high):
        self.patches = patches_chw
        self.canny_sigma = canny_sigma
        self.canny_low = canny_low
        self.canny_high = canny_high

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]  # (C,H,W) in [0,1]
        gray = compute_ndvi_from_patch(patch)
        edges = canny_edges(gray, sigma=self.canny_sigma, low_threshold=self.canny_low, high_threshold=self.canny_high)  # (H,W) in {0,1}

        x = torch.from_numpy(patch).float()
        y = torch.from_numpy(edges[None, :, :]).float()  # (1,H,W)
        return x, y


def read_tile_tif(filepath):
    """
    Reads a multi-band GeoTIFF tile into a numpy array.

    Args:
        filepath (str): Path to the GeoTIFF file.

    Returns:
        np.ndarray: Array of shape (C, H, W).
    """
    with rasterio.open(filepath) as src:
        arr = src.read()
    return arr


def list_tile_files(dataset_dir):
    """
    Lists training tile files (tile_001..tile_015) and the Teseachi tile.

    Args:
        dataset_dir (str): Folder containing GeoTIFF tiles.

    Returns:
        tuple[list[tuple[int,str]], str]: (train_tiles, teseachi_tile)
            train_tiles: list of (tile_id, filepath)
            teseachi_tile: filepath
    """
    train_tiles = []
    for i in range(1, 16):
        fname = f"tile_{i:03d}.tif"
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing tile file: {fpath}")
        train_tiles.append((i, fpath))

    teseachi_path = os.path.join(dataset_dir, "tile_teseachi.tif")
    if not os.path.exists(teseachi_path):
       raise FileNotFoundError(f"Missing Teseachi tile file: {teseachi_path}")

    return train_tiles, teseachi_path


def load_proxy_csv(proxy_csv_path):
    """
    Loads tile-level proxy biomass values from CSV.

    Expected columns: id, coordinate_x, coordinate_y, biomass

    Args:
        proxy_csv_path (str): Path to proxy_biomass.csv.

    Returns:
        dict: Mapping tile_id (int) -> biomass (float), and "Teseachi" -> biomass (float) if present.
    """
    proxy = {}
    with open(proxy_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["id"]
            biomass = float(row["biomass"])
            if tid.strip().lower() == "teseachi" or tid.strip().lower() == "teseachi":
                proxy["Teseachi"] = biomass
            else:
                proxy[int(tid)] = biomass
    return proxy


def build_training_patches(train_tiles, patch_size, stride):
    """
    Builds normalized patches for Stage A training from training tiles.

    Args:
        train_tiles (list[tuple[int,str]]): List of (tile_id, filepath).
        patch_size (int): Patch size.
        stride (int): Patch stride.

    Returns:
        tuple[list[np.ndarray], dict]: (patches, tile_patch_index)
            patches: list of arrays (C, patch_size, patch_size)
            tile_patch_index: tile_id -> list of patch indices in 'patches'
    """
    patches = []
    tile_patch_index = {}

    for tile_id, fpath in train_tiles:
        tile = read_tile_tif(fpath)
        tile = normalize_per_band(tile)

        tile_patches, _ = extract_patches(tile, patch_size=patch_size, stride=stride)
        tile_patch_index[tile_id] = list(range(len(patches), len(patches) + len(tile_patches)))
        patches.extend(tile_patches)

    return patches, tile_patch_index


def build_teseachi_patches(teseachi_path, patch_size, stride):
    """
    Builds normalized patches for Teseachi evaluation tile.

    Args:
        teseachi_path (str): Path to Teseachi GeoTIFF.
        patch_size (int): Patch size.
        stride (int): Patch stride.

    Returns:
        list[np.ndarray]: List of patches (C, patch_size, patch_size).
    """
    tile = read_tile_tif(teseachi_path)
    tile = normalize_per_band(tile)
    patches, _ = extract_patches(tile, patch_size=patch_size, stride=stride)
    return patches


# ============================================================
# We define training, embedding extraction, and plots/metrics
# ============================================================

def train_stage_a_ssl(model, loader, device, epochs, lr, plot_dir, weight_decay, pos_weight, grad_clip_max_norm, f1_threshold):
    """
    Trains the self-supervised edge prediction task (Stage A).

    Args:
        model (nn.Module): SSL model (encoder + decoder).
        loader (DataLoader): Training loader yielding (x_patch, y_edge).
        device (str): Device string ("cpu" or "cuda").
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        plot_dir (str): Output folder for plots.
        weight_decay (float): Adam weight decay.
        pos_weight (float): Positive class weight for BCEWithLogitsLoss (edge sparsity handling).
        grad_clip_max_norm (float): Gradient clipping max norm (set <=0 to disable).
        f1_threshold (float): Probability threshold for F1 computation.

    Returns:
        dict: Training history with keys: "loss", "f1".
    """
    os.makedirs(plot_dir, exist_ok=True)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # We address extreme class imbalance in edge maps (mostly zeros) using pos_weight.
    _pos_w = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=_pos_w)

    loss_hist = []
    f1_hist = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_losses = []
        ep_f1s = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            _, logits = model(xb)
            loss = crit(logits, yb)

            opt.zero_grad()
            loss.backward()
            if grad_clip_max_norm is not None and float(grad_clip_max_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_max_norm))
            opt.step()

            ep_losses.append(loss.item())
            ep_f1s.append(f1_for_edges(logits.detach(), yb.detach(), threshold=f1_threshold))

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        mean_f1 = float(np.mean(ep_f1s)) if ep_f1s else float("nan")

        loss_hist.append(mean_loss)
        f1_hist.append(mean_f1)

        print(f"[Stage A] Epoch {ep}/{epochs} | BCE={mean_loss:.6f} | F1={mean_f1:.4f}")

    # We plot training curves.
    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogitsLoss (edge maps)")
    plt.title("Stage A: Self-supervised loss")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "stage_a_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(f1_hist)
    plt.xlabel("Epoch")
    plt.ylabel("F1 (edge pixels)")
    plt.title("Stage A: Edge quality")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "stage_a_f1.png"))
    plt.close()

    return {"loss": loss_hist, "f1": f1_hist}


def visualize_stage_a_samples(model, patches, device, plot_dir, canny_sigma, canny_low, canny_high, n=3, pred_threshold=0.25):
    """
    Plots a few sample inputs with their edge targets and model predictions.

    Args:
        model (nn.Module): Trained SSL model.
        patches (list[np.ndarray]): List of normalized patches (C,H,W).
        device (str): Device string ("cpu" or "cuda").
        plot_dir (str): Output folder for plots.
        canny_sigma (float): Canny smoothing parameter (sigma).
        canny_low (float): Lower hysteresis threshold (0..1).
        canny_high (float): Upper hysteresis threshold (0..1).
        n (int): Number of samples to plot.
        pred_threshold (float): Probability threshold for predicted edges.

    Returns:
        None: This function saves figures.
    """
    os.makedirs(plot_dir, exist_ok=True)
    model.eval()
    model = model.to(device)

    idxs = list(range(min(n, len(patches))))
    for k, idx in enumerate(idxs, start=1):
        patch = patches[idx]
        gray = compute_ndvi_from_patch(patch)
        target = canny_edges(gray, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)

        xb = torch.from_numpy(patch[None, :, :, :]).float().to(device)
        with torch.no_grad():
            _, logits = model(xb)
            pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= float(pred_threshold)).astype(np.uint8)

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(gray, cmap="gray")
        plt.title("NDVI-like (for edge target)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(target, cmap="gray")
        plt.title(f"Target edges (sigma={canny_sigma}, low={canny_low}, high={canny_high})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred, cmap="gray")
        plt.title(f"Predicted edges (thr={pred_threshold})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"stage_a_sample_{k}.png"))
        plt.close()


def extract_patch_embeddings(encoder, patches, device, batch_size):
    """
    Extracts encoder embeddings for a list of patches.

    Args:
        encoder (nn.Module): Trained encoder.
        patches (list[np.ndarray]): List of normalized patches (C,H,W).
        device (str): Device string.
        batch_size (int): Batch size for embedding extraction.

    Returns:
        np.ndarray: Embeddings of shape (N, D).
    """
    encoder.eval()
    encoder = encoder.to(device)

    z_all = []
    n = len(patches)
    i = 0
    while i < n:
        batch = patches[i:i + batch_size]
        xb = torch.from_numpy(np.stack(batch, axis=0)).float().to(device)
        with torch.no_grad():
            z = encoder(xb).cpu().numpy()
        z_all.append(z)
        i += batch_size

    return np.vstack(z_all) if z_all else np.zeros((0, 64), dtype=np.float32)


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
    X_list = []
    y_list = []

    for tid in train_tile_ids:
        z = embeddings_by_tile[tid]
        if tid not in proxy_map:
            raise KeyError(f"Missing proxy value for tile id {tid} in proxy_biomass.csv")

        y_tile = float(proxy_map[tid])
        y_rep = np.full((z.shape[0],), y_tile, dtype=np.float32)

        X_list.append(z.astype(np.float32))
        y_list.append(y_rep)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def train_stage_b_ridge(X_train, y_train, alpha):
    """
    Trains Ridge regression for Stage B (embedding -> biomass proxy).

    Args:
        X_train (np.ndarray): Training embeddings of shape (N, D).
        y_train (np.ndarray): Training targets of shape (N,).
        alpha (float): Ridge regularization strength.

    Returns:
        object: Trained sklearn pipeline (StandardScaler -> Ridge).
    """
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True, random_state=0))
    model.fit(X_train, y_train)
    return model


def evaluate_stage_b(y_true, y_pred):
    """
    Computes Stage B evaluation metrics (R2, RMSE, Spearman).

    Args:
        y_true (np.ndarray): True biomass values of shape (N,).
        y_pred (np.ndarray): Predicted biomass values of shape (N,).

    Returns:
        dict: Metrics dictionary with keys: "r2", "rmse", "spearman".
    """
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    rho = float(spearman_corr(y_pred, y_true))
    return {"r2": r2, "rmse": rmse, "spearman": rho}


def plot_stage_b_scatter(y_true, y_pred, plot_path, title):
    """
    Plots predicted vs true scatter for Stage B.

    Args:
        y_true (np.ndarray): True values of shape (N,).
        y_pred (np.ndarray): Predicted values of shape (N,).
        plot_path (str): Output path for the figure.
        title (str): Plot title.

    Returns:
        None: This function saves a figure.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.xlabel("True biomass")
    plt.ylabel("Predicted biomass")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


# ============================================================
# We define a main runner in the exact pipeline order
# ============================================================

def main():
    t0_all = time.time()

    dataset_dir = "dataset"
    proxy_dir = "proxy_biomass"
    proxy_csv_path = os.path.join(proxy_dir, "proxy_biomass.csv")

    output_dir = "outputs"
    stage_a_plot_dir = os.path.join(output_dir, "stage_a_plots")
    stage_b_plot_dir = os.path.join(output_dir, "stage_b_plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stage_a_plot_dir, exist_ok=True)
    os.makedirs(stage_b_plot_dir, exist_ok=True)

    # Hyperparameters (updated)
    patch_size = 96
    stride = 48

    canny_sigma = 1.5
    canny_low = 0.05
    canny_high = 0.15

    batch_size = 128
    epochs = 30
    lr = 3e-4
    weight_decay = 1e-4
    grad_clip_max_norm = 1.0

    pos_weight = 25.0
    f1_threshold = 0.25
    pred_threshold = 0.25

    ridge_alpha = 10.0

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # ============================================================
    # 1) We load tiles and build patches (Stage A input X)
    # ============================================================
    t0 = time.time()
    print("\n[Subsection] Loading tiles and extracting patches (Stage A: Images input X)")

    train_tiles, teseachi_path = list_tile_files(dataset_dir)
    if not os.path.exists(proxy_csv_path):
        raise FileNotFoundError(f"Missing proxy CSV: {proxy_csv_path}")

    proxy_map = load_proxy_csv(proxy_csv_path)

    train_patches, train_tile_patch_index = build_training_patches(train_tiles=train_tiles, patch_size=patch_size, stride=stride)
    teseachi_patches = build_teseachi_patches(teseachi_path=teseachi_path, patch_size=patch_size, stride=stride)

    # ============================
    # (B) REPLACEMENT PRINT BLOCK
    # ============================
    # We infer expected patch counts from the actual tile raster sizes on disk.
    with rasterio.open(train_tiles[0][1]) as _src0:
        h0, w0 = _src0.height, _src0.width
    n_h0, n_w0, per_tile_expected = expected_patches_per_tile(h0, w0, patch_size, stride)
    total_expected = per_tile_expected * len(train_tiles)

    with rasterio.open(teseachi_path) as _srcT:
        hT, wT = _srcT.height, _srcT.width
    n_hT, n_wT, teseachi_expected = expected_patches_per_tile(hT, wT, patch_size, stride)

    print(f"We found {len(train_tiles)} training tiles and 1 Teseachi evaluation tile.")
    print(f"We extracted {len(train_patches)} training patches total (expected {len(train_tiles)} * {per_tile_expected} = {total_expected}; tile size {h0}x{w0} -> {n_h0}x{n_w0} patches).")
    print(f"We extracted {len(teseachi_patches)} Teseachi patches (expected {teseachi_expected}; tile size {hT}x{wT} -> {n_hT}x{n_wT} patches).")

    print(f"[Time] Loading/patch extraction elapsed: {format_seconds(time.time() - t0)}")

    # ============================================================
    # 2) We train Stage A (SSL)
    # ============================================================
    t0 = time.time()
    print("\n[Subsection] Stage A: Self-supervised Tiny CNN (Canny edge task)")

    ssl_dataset = PatchSslDataset(patches_chw=train_patches, canny_sigma=canny_sigma, canny_low=canny_low, canny_high=canny_high)
    ssl_loader = DataLoader(ssl_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    in_channels = train_patches[0].shape[0]
    ssl_model = SslEdgeModel(in_channels=in_channels, emb_dim=64, patch_size=patch_size)

    hist = train_stage_a_ssl(model=ssl_model, loader=ssl_loader, device=device, epochs=epochs, lr=lr, plot_dir=stage_a_plot_dir, weight_decay=weight_decay, pos_weight=pos_weight, grad_clip_max_norm=grad_clip_max_norm, f1_threshold=f1_threshold)

    visualize_stage_a_samples(model=ssl_model, patches=train_patches, device=device, plot_dir=stage_a_plot_dir, canny_sigma=canny_sigma, canny_low=canny_low, canny_high=canny_high, n=3, pred_threshold=pred_threshold)

    print(f"[Time] Stage A training elapsed: {format_seconds(time.time() - t0)}")

    # ============================================================
    # 3) We extract embeddings (Stage B features)
    # ============================================================
    t0 = time.time()
    print("\n[Subsection] Extracting patch embeddings (Learned Representations)")

    embeddings_by_tile = {}
    encoder = ssl_model.encoder

    for tid, _ in train_tiles:
        idxs = train_tile_patch_index[tid]
        tile_patches = [train_patches[i] for i in idxs]
        z = extract_patch_embeddings(encoder=encoder, patches=tile_patches, device=device, batch_size=batch_size)
        embeddings_by_tile[tid] = z

    z_teseachi = extract_patch_embeddings(encoder=encoder, patches=teseachi_patches, device=device, batch_size=batch_size)

    print(f"We extracted embeddings for 15 tiles (train) and 1 tile (Teseachi).")
    print(f"[Time] Embedding extraction elapsed: {format_seconds(time.time() - t0)}")

    # ============================================================
    # 4) Stage B: Ridge regression
    # ============================================================
    t0 = time.time()
    print("\n[Subsection] Stage B: Proxy regression (Ridge)")

    train_tile_ids = [tid for tid, _ in train_tiles]
    X_train, y_train = build_stage_b_training_data(embeddings_by_tile=embeddings_by_tile, proxy_map=proxy_map, train_tile_ids=train_tile_ids)

    ridge = train_stage_b_ridge(X_train=X_train, y_train=y_train, alpha=ridge_alpha)
    yhat_teseachi = ridge.predict(z_teseachi).astype(np.float32)

    print(f"We trained Ridge on X shape {X_train.shape} with y shape {y_train.shape}.")
    print(f"We produced Teseachi patch predictions with shape {yhat_teseachi.shape}.")
    print(f"[Time] Stage B training/prediction elapsed: {format_seconds(time.time() - t0)}")

    # ============================================================
    # 5) Metrics + plots
    # ============================================================
    t0 = time.time()
    print("\n[Subsection] Metrics + plots")

    stage_a_loss_last = hist["loss"][-1] if hist["loss"] else float("nan")
    stage_a_f1_last = hist["f1"][-1] if hist["f1"] else float("nan")

    print(f"[Stage A Metric] Final loss (lower is better): {stage_a_loss_last:.6f}")
    print(f"[Stage A Metric] Final F1 (higher is better): {stage_a_f1_last:.4f}")

    teseachi_truth_path = os.path.join(proxy_dir, "teseachi_measured_biomass.csv")

    if os.path.exists(teseachi_truth_path):
        y_true_teseachi, yhat_use = load_teseachi_truth_and_align(teseachi_truth_csv_path=teseachi_truth_path, yhat_teseachi=yhat_teseachi)

        m = evaluate_stage_b(y_true=y_true_teseachi, y_pred=yhat_use)

        print(f"[Stage B Metric] R2 on Teseachi (higher is better): {m['r2']:.4f}")
        print(f"[Stage B Metric] RMSE on Teseachi (lower is better): {m['rmse']:.4f}")
        print(f"[Stage B Metric] Spearman rho on Teseachi (higher is better): {m['spearman']:.4f}")

        plot_stage_b_scatter(y_true=y_true_teseachi, y_pred=yhat_use, plot_path=os.path.join(stage_b_plot_dir, "stage_b_teseachi_scatter.png"), title="Stage B: Predicted vs proxy biomass (Teseachi)")
    else:
        print("[Stage B Metric] Proxy Teseachi biomass file not found, so R2/RMSE/Spearman vs proxy cannot be computed.")
        print(f"We expected: {teseachi_truth_path}")
        print("We still saved Stage A plots and produced Teseachi predictions (yhat).")

        plt.figure()
        plt.hist(yhat_teseachi, bins=30)
        plt.xlabel("Predicted biomass (proxy scale)")
        plt.ylabel("Count")
        plt.title("Stage B: Teseachi predicted biomass distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(stage_b_plot_dir, "stage_b_teseachi_pred_hist.png"))
        plt.close()

    print(f"[Time] Metrics/plots elapsed: {format_seconds(time.time() - t0)}")
    print(f"\n[Time] Total elapsed: {format_seconds(time.time() - t0_all)}")


if __name__ == "__main__":
    main()
