# Data Wrangler. 
import os
import json
import time

import rasterio
import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient

from utilities.utilities import *

def tile_pixels_per_side(tile_size_km, target_res_m):
    """
    Computes the pixel size per side for a square tile.

    Args:
        tile_size_km (float): Tile size in kilometers (one side).
        target_res_m (float): Target meters per pixel.

    Returns:
        int: Pixels per side (width and height).
    """
    meters = float(tile_size_km) * 1000.0
    return max(1, int(meters / float(target_res_m) + 0.5))


def s2_tiff(oauth, process_url, bbox, time_from, time_to, width_px, height_px, max_cloud_coverage, mosaicking_order, upsampling, downsampling):
    """
    Requests sentinel-2 l2a bands for a bbox and time range and returns a tiff in memory.

    Args:
        oauth (OAuth2Session): Authenticated session.
        process_url (str): Process api url.
        bbox (list): [lon_min, lat_min, lon_max, lat_max].
        time_from (str): Iso timestamp start.
        time_to (str): Iso timestamp end.
        width_px (int): Output width.
        height_px (int): Output height.
        max_cloud_coverage (int): Max cloud coverage percent.
        mosaicking_order (str): Mosaicking order string.
        upsampling (str): Upsampling mode.
        downsampling (str): Downsampling mode.

    Returns:
        tuple: (data, transform, crs) where data is (6,h,w) float32.
    """
    evalscript = """//VERSION=3
function setup() {
  return {
    input: ["B03","B04","B05","B06","B07","B08"],
    output: { bands: 6, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(s) {
  return [
    s.B08,
    s.B04,
    s.B05,
    s.B06,
    s.B07,
    s.B03
  ];
}
"""
    body = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "bbox": bbox,
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": time_from, "to": time_to},
                        "maxCloudCoverage": int(max_cloud_coverage),
                        "mosaickingOrder": str(mosaicking_order),
                    },
                    "processing": {
                        "upsampling": str(upsampling),
                        "downsampling": str(downsampling),
                    },
                }
            ],
        },
        "output": {
            "width": int(width_px),
            "height": int(height_px),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    resp = oauth.post(process_url, json=body)
    if not resp.ok:
        raise RuntimeError(f"s2 request failed: {resp.status_code}\n{resp.text}")

    with MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            data = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs

    return data, transform, crs


def dem_tiff(oauth, process_url, bbox, width_px, height_px, dem_instance, upsampling, downsampling):
    """
    Requests dem for a bbox and returns a tiff in memory.

    Args:
        oauth (OAuth2Session): Authenticated session.
        process_url (str): Process api url.
        bbox (list): [lon_min, lat_min, lon_max, lat_max].
        width_px (int): Output width.
        height_px (int): Output height.
        dem_instance (str): Dem instance name.
        upsampling (str): Upsampling mode.
        downsampling (str): Downsampling mode.

    Returns:
        tuple: (dem_data, transform, crs) where dem_data is (h,w) float32.
    """
    evalscript = """//VERSION=3
function setup() {
  return { input: ["DEM"], output: { id: "default", bands: 1, sampleType: SampleType.FLOAT32 } };
}
function evaluatePixel(s) { return [s.DEM]; }
"""
    body = {
        "input": {
            "bounds": {
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
                "bbox": bbox,
            },
            "data": [
                {
                    "type": "dem",
                    "dataFilter": {"demInstance": str(dem_instance)},
                    "processing": {"upsampling": str(upsampling), "downsampling": str(downsampling)},
                }
            ],
        },
        "output": {
            "width": int(width_px),
            "height": int(height_px),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }

    resp = oauth.post(process_url, json=body)
    if not resp.ok:
        raise RuntimeError(f"dem request failed: {resp.status_code}\n{resp.text}")

    with MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            data = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs

    return data[0], transform, crs


def write_multiband_geotiff(path, data, transform, crs, compress):
    """
    Writes a multi-band geotiff with compression.

    Args:
        path (str): Output path.
        data (np.ndarray): Array shaped (bands, h, w).
        transform (Affine): Geo transform.
        crs: Rasterio crs.
        compress (str): Compression name.

    Returns:
        None
    """
    bands, height, width = data.shape
    os.makedirs(os.path.dirname(path), exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": int(bands),
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": str(compress),
    }

    with rasterio.open(path, "w", **profile) as dst:
        for i in range(bands):
            dst.write(data[i], i + 1)


def main_data_wrangler():
    """
    Downloads sentinel-2 + dem tiles and saves them as compressed geotiffs.

    Args:
        None

    Returns:
        None
    """
    t0 = time.time()
    cfg = read_json("utilities/configuration.json")

    token_url = cfg["auth"]["token_url"]
    process_url = cfg["auth"]["process_url"]
    client_id = cfg["auth"]["client_id_env"]
    client_secret = cfg["auth"]["client_secret_env"]

    tile_size_km = cfg["tiles"]["tile_size_km"]
    tile_defs = cfg["tiles"]["tile_defs"]
    teseachi_bbox = cfg["tiles"]["teseachi_bbox"]

    dw = cfg["data_wrangler"]
    time_from = dw["time_from"]
    time_to = dw["time_to"]
    target_res_m = dw["target_res_m"]
    max_cloud_coverage = dw["max_cloud_coverage"]
    mosaicking_order = dw["mosaicking_order"]
    dem_instance = dw["dem_instance"]
    upsampling = dw["upsampling"]
    downsampling = dw["downsampling"]
    out_dir = dw["out_dir"]
    compress = dw["compress"]
    tile_filename_template = dw["tile_filename_template"]
    teseachi_filename = dw["teseachi_filename"]

    print("[auth] building oauth session...")
    oauth = build_oauth_session(client_id, client_secret, token_url)

    pixels = tile_pixels_per_side(tile_size_km, target_res_m)
    width_px = pixels
    height_px = pixels

    os.makedirs(out_dir, exist_ok=True)

    all_tiles = [{"id": int(t["id"]), "bbox": t["bbox"]} for t in tile_defs] + [{"id": 16, "bbox": teseachi_bbox}]

    for i, t in enumerate(all_tiles, start=1):
        tile_id = int(t["id"])
        bbox = t["bbox"]

        filename = teseachi_filename if tile_id == 16 else tile_filename_template.format(id=tile_id)
        out_path = os.path.join(out_dir, filename)

        # We print what we process.
        print(f"[tile {i:02d}/{len(all_tiles):02d}] downloading tile_id={tile_id} bbox={bbox} -> {out_path}")

        s2_data, s2_transform, s2_crs = s2_tiff(oauth=oauth, process_url=process_url, bbox=bbox, time_from=time_from, time_to=time_to, width_px=width_px, height_px=height_px, max_cloud_coverage=max_cloud_coverage, mosaicking_order=mosaicking_order, upsampling=upsampling, downsampling=downsampling)

        dem_data, dem_transform, dem_crs = dem_tiff(oauth=oauth, process_url=process_url, bbox=bbox, width_px=width_px, height_px=height_px, dem_instance=dem_instance, upsampling=upsampling, downsampling=downsampling)

        if s2_data.shape[1:] != dem_data.shape:
            raise RuntimeError(f"shape mismatch tile_id={tile_id}: s2={s2_data.shape} dem={dem_data.shape}")
        if s2_crs != dem_crs:
            raise RuntimeError(f"crs mismatch tile_id={tile_id}: s2={s2_crs} dem={dem_crs}")

        dem_band = dem_data[np.newaxis, :, :]
        merged = np.concatenate([s2_data, dem_band], axis=0)

        write_multiband_geotiff(out_path, merged, s2_transform, s2_crs, compress)
        print(f"[ok] file has been successfully downloaded: {out_path}")

    elapsed = time.time() - t0
    print(f"[done] time elapsed: {format_seconds(elapsed)}")
