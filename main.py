import os
import time
import torch

from utilities.utilities import *

from modeling.training import *
from modeling.data_wrangler import * 

from quality.quality_vegetation import * 
from quality.quality_geographical_tiles import * 

# ============================================================
# We define a main runner in the exact pipeline order
# ============================================================
def main():
    t0_all = time.time()

    dataset_dir = "dataset\\tiles"
    proxy_dir = "dataset\\proxy_biomass"
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
    epochs = 100
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

