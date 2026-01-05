# Vegetation table
import io
import os
import json
import time

import requests

import zipfile
import shutil
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
 
import rasterio
from rasterio.io import MemoryFile

from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient

from utilities.utilities import *


def filter_natural_pasture(gdf, keywords_lower):
    """
    Filters features whose string columns match the configured pasture keywords.

    Args:
        gdf (GeoDataFrame): Input geodataframe.
        keywords_lower (list): Lowercased keywords to search.

    Returns:
        GeoDataFrame: Filtered geodataframe.
    """
    if gdf.empty:
        return gdf

    mask = np.zeros(len(gdf), dtype=bool)
    for col in gdf.columns:
        if gdf[col].dtype == object:
            vals = gdf[col].astype(str).str.lower()
            col_mask = np.zeros(len(gdf), dtype=bool)
            for kw in keywords_lower:
                col_mask |= vals.str.contains(str(kw), na=False)
            mask |= col_mask

    return gdf[mask].copy()


def load_inegi_natural_pasture(shp_paths, keywords_lower):
    """
    Loads inegi shapefiles and filters them to natural pastures.

    Args:
        shp_paths (list): List of shapefile paths (can be relative filenames).
        keywords_lower (list): Lowercased keywords to search.

    Returns:
        GeoDataFrame: Merged pasture polygons in epsg:4326.
    """
    repo_out_dir = "special_repo_for_shape"
    repo_zip_url = "https://github.com/amcm329/special_repo_for_shape/archive/refs/heads/main.zip"

    os.makedirs(repo_out_dir, exist_ok=True)

    with urllib.request.urlopen(repo_zip_url) as r:
        with zipfile.ZipFile(io.BytesIO(r.read())) as z:
            z.extractall(repo_out_dir)

    extracted_root = os.path.join(repo_out_dir, "special_repo_for_shape-main")
    shp_file = os.path.join(extracted_root, "chihuahua_tiles.shp")

    try:
        print(f"[inegi] reading {shp_file} from downloaded repo...")

        g = gpd.read_file(shp_file)
        if g.crs is None:
            raise RuntimeError("chihuahua_tiles has no CRS")

        g = g.to_crs("EPSG:4326")

        print(f"[inegi] file=chihuahua_tiles total={len(g)} (no filtering applied)")
        return g

    finally:
        shutil.rmtree(repo_out_dir, ignore_errors=True)


def pasture_fraction_for_tile(tile_poly, pasture_eq, eq_area_crs):
    """
    Computes the fraction of tile area covered by natural pasture polygons.

    Args:
        tile_poly (shapely geometry): Tile polygon in epsg:4326.
        pasture_eq (GeoDataFrame): Pasture polygons in an equal-area crs.
        eq_area_crs (str): Equal-area crs string.

    Returns:
        float: Pasture fraction in [0, 1].
    """
    tile_eq = gpd.GeoDataFrame({"id": [0]}, geometry=[tile_poly], crs="EPSG:4326").to_crs(eq_area_crs)
    tile_geom_eq = tile_eq.geometry.iloc[0]
    tile_area = float(tile_geom_eq.area)

    inter = gpd.overlay(pasture_eq, tile_eq, how="intersection")
    if inter.empty or tile_area <= 0:
        return 0.0

    past_area = float(inter.geometry.area.sum())
    return float(past_area / tile_area)


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
        np.ndarray: Dem array shaped (h,w) float32.
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

    return data[0]


def dem_mean_for_bbox(oauth, process_url, bbox, coarse_px, dem_instance, upsampling, downsampling):
    """
    Computes a coarse mean dem for a bbox.

    Args:
        oauth (OAuth2Session): Authenticated session.
        process_url (str): Process api url.
        bbox (list): [lon_min, lat_min, lon_max, lat_max].
        coarse_px (int): Coarse width and height.
        dem_instance (str): Dem instance name.
        upsampling (str): Upsampling mode.
        downsampling (str): Downsampling mode.

    Returns:
        float: Mean dem value.
    """
    dem_data = dem_tiff(oauth, process_url, bbox, coarse_px, coarse_px, dem_instance, upsampling, downsampling)
    return float(np.nanmean(dem_data))


def main_pasture():
    """
    Computes pasture fractions and dem summaries and saves them as a csv.

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

    state = load_state_polygon(cfg)
    state_union = state.unary_union

    pasture_cfg = cfg["pasture"]
    shp_paths = pasture_cfg["inegi_shp_paths"]
    eq_area_crs = pasture_cfg["eq_area_crs"]
    keywords_lower = pasture_cfg["keywords_lower"]
    out_csv = pasture_cfg["out_csv"]
    dem_coarse_px = pasture_cfg["dem_coarse_px"]

    dw = cfg["data_wrangler"]
    dem_instance = dw["dem_instance"]
    upsampling = dw["upsampling"]
    downsampling = dw["downsampling"]

    print("[auth] building oauth session...")
    oauth = build_oauth_session(client_id, client_secret, token_url)

    print("[inegi] loading pasturees naturales...")
    pasture_all = load_inegi_natural_pasture(shp_paths, keywords_lower)
    pasture_eq = pasture_all.to_crs(eq_area_crs)

    tile_defs = cfg["tiles"]["tile_defs"]
    teseachi_bbox = cfg["tiles"]["teseachi_bbox"]

    rows = []
    print("[tiles] evaluating 15 tiles + teseachi...")

    # We do sampling tiles.
    for t in tile_defs:
        tid = int(t["id"])
        bbox = t["bbox"]
        tile_poly = box(*bbox)
        inside = bool(tile_poly.intersects(state_union))

        frac_past = pasture_fraction_for_tile(tile_poly, pasture_eq, eq_area_crs)
        mean_dem = dem_mean_for_bbox(oauth, process_url, bbox, dem_coarse_px, dem_instance, upsampling, downsampling)

        rows.append(
            {
                "id": tid,
                "bbox": bbox,
                "inside_chihuahua": inside,
                "pasture_frac": float(frac_past),
                "mean_dem": float(mean_dem),
            }
        )

    # We do teseachi.
    tes_poly = box(*teseachi_bbox)
    tes_inside = bool(tes_poly.intersects(state_union))
    tes_frac = pasture_fraction_for_tile(tes_poly, pasture_eq, eq_area_crs)
    tes_mean_dem = dem_mean_for_bbox(oauth, process_url, teseachi_bbox, dem_coarse_px, dem_instance, upsampling, downsampling)

    rows.append(
        {
            "id": 16,
            "bbox": teseachi_bbox,
            "inside_chihuahua": tes_inside,
            "pasture_frac": float(tes_frac),
            "mean_dem": float(tes_mean_dem),
        }
    )

    df = pd.DataFrame(rows).sort_values("id")
    print(df.to_string(index=False))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[ok] file has been successfully saved: {out_csv}")

    elapsed = time.time() - t0
    print(f"[done] time elapsed: {format_seconds(elapsed)}")
