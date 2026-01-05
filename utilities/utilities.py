# We define Utilities (generic functions relevant to all sections in this code).
import os
import csv
import rasterio
import numpy as np

from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient


def format_seconds(seconds):
    """
    Formats elapsed seconds into a human-readable string.

    Args:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds - 60 * minutes
    return f"{minutes}m {rem:.2f}s"


def read_json(path):
    """
    Reads a json file from disk and returns the parsed object.

    Args:
        path (str): Path to a json file.

    Returns:
        dict: Parsed json as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_oauth_session(client_id, client_secret, token_url):
    """
    Builds an oauth2 session for copernicus data space.

    Args:
        client_id (str): OAuth client id.
        client_secret (str): OAuth client secret.
        token_url (str): Token url.

    Returns:
        OAuth2Session: Authenticated session.
    """
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    oauth.fetch_token(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        include_client_id=True,
    )
    return oauth


def load_gadm_level1_in_memory(gadm_zip, level1_member):
    """
    Loads the gadm level-1 shapefile directly from a remote zip without using gdal.

    Args:
        gadm_zip (str): Url to gadm zip.
        level1_member (str): Member shapefile name inside the zip.

    Returns:
        GeoDataFrame: Level-1 gadm geodataframe.
    """
    # We read the shapefile directly from the remote zip using geopandas.
    gdf = gpd.read_file(f"zip+{gadm_zip}!{level1_member}")
    return gdf


def load_state_polygon(cfg):
    """
    Loads the target state polygon from gadm.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        GeoDataFrame: State polygon in epsg:4326.
    """
    gadm_zip = cfg["gadm"]["gadm_zip"]
    level1_member = cfg["gadm"]["level1_member"]
    state_name_field = cfg["gadm"]["state_name_field"]
    state_name_value = cfg["gadm"]["state_name_value"]

    mex = load_gadm_level1_in_memory(gadm_zip, level1_member)
    state = mex[mex[state_name_field] == state_name_value].to_crs("EPSG:4326")
    if state.empty:
        raise RuntimeError(f"could not find state={state_name_value} in gadm level-1")
    return state


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
