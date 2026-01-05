# For Geographical tiles.
import json
import math
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import geopandas as gpd
from shapely.geometry import box, Polygon, MultiPolygon

import rasterio
from rasterio.io import MemoryFile
from rasterio.features import rasterize

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import requests
from osgeo import gdal

from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


def bbox_to_pixels(bbox, res_m):
    """
    Converts a geographic bbox into approximate pixel dimensions at a target meter resolution.

    Args:
        bbox (list): [lon_min, lat_min, lon_max, lat_max].
        res_m (float): Target meters per pixel.

    Returns:
        tuple: (width_px, height_px, width_m, height_m).
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    mean_lat = 0.5 * (lat_min + lat_max)

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mean_lat))

    width_m = (lon_max - lon_min) * m_per_deg_lon
    height_m = (lat_max - lat_min) * m_per_deg_lat

    width_px = max(1, int(width_m / float(res_m)))
    height_px = max(1, int(height_m / float(res_m)))

    return width_px, height_px, float(width_m), float(height_m)


def request_state_rgb(oauth, process_url, bbox, width_px, height_px, time_from, time_to, max_cloud_coverage, mosaicking_order):
    """
    Requests a sentinel-2 rgb mosaic for a bbox and returns it as an array.

    Args:
        oauth (OAuth2Session): Authenticated session.
        process_url (str): Process api url.
        bbox (list): [lon_min, lat_min, lon_max, lat_max].
        width_px (int): Output width.
        height_px (int): Output height.
        time_from (str): Iso timestamp start.
        time_to (str): Iso timestamp end.
        max_cloud_coverage (int): Max cloud coverage percent.
        mosaicking_order (str): Mosaicking order string.

    Returns:
        tuple: (rgb, transform, crs) where rgb is (h,w,3) float32.
    """
    evalscript = """//VERSION=3
function setup() {
  return { input: ["B04","B03","B02"], output: { bands: 3, sampleType: "FLOAT32" } };
}
function evaluatePixel(s) { return [s.B04, s.B03, s.B02]; }
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
        raise RuntimeError(f"state rgb request failed: {resp.status_code}\n{resp.text}")

    with MemoryFile(resp.content) as memfile:
        with memfile.open() as src:
            data = src.read().astype(np.float32)  # (3,h,w)
            transform = src.transform
            crs = src.crs

    rgb = np.transpose(data, (1, 2, 0))
    return rgb, transform, crs


def normalize_rgb(rgb, mask):
    """
    Normalizes rgb values to [0,1] using robust percentiles over masked pixels.

    Args:
        rgb (np.ndarray): Rgb array shaped (h,w,3).
        mask (np.ndarray): Mask shaped (h,w), 1 inside, 0 outside.

    Returns:
        np.ndarray: Normalized rgb array shaped (h,w,3).
    """
    rgb_masked = rgb.copy()
    for c in range(3):
        channel = rgb_masked[..., c]
        channel[mask == 0] = np.nan
        rgb_masked[..., c] = channel

    flat = rgb_masked.reshape(-1, 3)
    flat = flat[~np.isnan(flat).any(axis=1)]
    if flat.size == 0:
        raise RuntimeError("no valid pixels found inside mask")

    vmin = float(np.percentile(flat, 2))
    vmax = float(np.percentile(flat, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    rgb_vis = (rgb_masked - vmin) / (vmax - vmin)
    rgb_vis = np.clip(rgb_vis, 0.0, 1.0)

    # We set outside-state pixels to white for a clean background.
    rgb_vis[np.isnan(rgb_vis)] = 1.0
    return rgb_vis


def main_vegetation(config_path, client_id, client_secret):
    """
    Creates a chihuahua map mosaic and overlays sampling tiles and teseachi bbox, then saves a png.

    Args:
        config_path (str): Path to configuration json.
        client_id (str): OAuth client id.
        client_secret (str): OAuth client secret.

    Returns:
        None
    """
    t0 = time.time()
    cfg = read_json(config_path)

    token_url = cfg["auth"]["token_url"]
    process_url = cfg["auth"]["process_url"]

    vp = cfg["vegetation_plot"]
    margin_deg = float(vp["margin_deg"])
    target_res_m = float(vp["target_res_m"])
    max_dim_px = int(vp["max_dim_px"])
    time_from = vp["time_from"]
    time_to = vp["time_to"]
    max_cloud_coverage = int(vp["max_cloud_coverage"])
    mosaicking_order = vp["mosaicking_order"]
    out_png = vp["out_png"]
    title = vp["title"]

    tile_defs = cfg["tiles"]["tile_defs"]
    teseachi_bbox = cfg["tiles"]["teseachi_bbox"]

    print("[auth] building oauth session...")
    oauth = build_oauth_session(client_id, client_secret, token_url)

    print("[gadm] loading chihuahua polygon in memory...")
    state = load_state_polygon(cfg)

    minx, miny, maxx, maxy = state.total_bounds
    roi_bbox = [float(minx - margin_deg), float(miny - margin_deg), float(maxx + margin_deg), float(maxy + margin_deg)]
    print(f"[roi] bbox_with_margin={roi_bbox}")

    width_px, height_px, width_m, height_m = bbox_to_pixels(roi_bbox, target_res_m)
    max_dim = max(width_px, height_px)
    if max_dim > max_dim_px:
        scale = float(max_dim_px) / float(max_dim)
        width_px = max(1, int(width_px * scale))
        height_px = max(1, int(height_px * scale))

    res_x_m = float(width_m) / float(width_px)
    res_y_m = float(height_m) / float(height_px)
    mean_res_m = 0.5 * (res_x_m + res_y_m)
    print(f"[mosaic] size={width_px}x{height_px} approx_res_m={mean_res_m:.1f}")

    print("[mosaic] requesting sentinel-2 rgb...")
    rgb, transform, crs = request_state_rgb(
        oauth=oauth,
        process_url=process_url,
        bbox=roi_bbox,
        width_px=width_px,
        height_px=height_px,
        time_from=time_from,
        time_to=time_to,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
    )

    # We rasterize the state polygon into the mosaic grid.
    state_r = state.to_crs(crs) if state.crs != crs else state
    shapes = [(geom, 1) for geom in state_r.geometry]
    mask = rasterize(shapes=shapes, out_shape=(rgb.shape[0], rgb.shape[1]), transform=transform, fill=0, dtype="uint8")

    rgb_vis = normalize_rgb(rgb, mask)

    # We build tiles geodataframes.
    tile_polys = [box(*t["bbox"]) for t in tile_defs]
    tiles_gdf = gpd.GeoDataFrame({"id": [int(t["id"]) for t in tile_defs]}, geometry=tile_polys, crs="EPSG:4326")
    teseachi_gdf = gpd.GeoDataFrame({"name": ["teseachi"]}, geometry=[box(*teseachi_bbox)], crs="EPSG:4326")

    proj = ccrs.PlateCarree()

    fig = plt.figure(figsize=(8, 10))
    fig.patch.set_facecolor("white")

    ax = plt.axes(projection=proj)
    ax.set_facecolor("white")

    extent_state = [roi_bbox[0], roi_bbox[2], roi_bbox[1], roi_bbox[3]]
    ax.imshow(rgb_vis, origin="upper", extent=extent_state, transform=proj)
    ax.set_title(title, fontsize=12)
    ax.set_extent(extent_state, crs=proj)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", color="gray")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # We draw the state outline.
    for geom in state.geometry:
        if isinstance(geom, Polygon):
            geoms = [geom]
        elif isinstance(geom, MultiPolygon):
            geoms = list(geom.geoms)
        else:
            geoms = []
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, color="black", linewidth=1.0, transform=proj)

    # We draw teseachi.
    for _, row in teseachi_gdf.iterrows():
        poly = row.geometry
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, facecolor="magenta", alpha=0.25, transform=proj)
        ax.plot(xs, ys, color="magenta", linewidth=2, transform=proj)

    # We draw sampling tiles.
    for _, row in tiles_gdf.iterrows():
        poly = row.geometry
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, facecolor="cyan", alpha=0.30, transform=proj)
        ax.plot(xs, ys, color="cyan", linewidth=1.5, transform=proj)

    tile_patch = mpatches.Patch(facecolor="cyan", edgecolor="cyan", alpha=0.30, label="sampling tiles (1–15)")
    tes_patch = mpatches.Patch(facecolor="magenta", edgecolor="magenta", alpha=0.25, label="teseachi tile")
    ax.legend(handles=[tile_patch, tes_patch], loc="lower right", frameon=True, fontsize=10)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[ok] vegetation plot has been successfully saved: {out_png}")

    elapsed = time.time() - t0
    print(f"[done] time elapsed: {format_seconds(elapsed)}")


if __name__ == "__main__":
    config_path = os.getenv("config_path", "configuration.json")
    cfg = read_json(config_path)
    client_id = os.getenv(cfg["auth"]["client_id_env"])
    client_secret = os.getenv(cfg["auth"]["client_secret_env"])
    if not client_id or not client_secret:
        raise RuntimeError("missing client_id/client_secret in environment")

    # CLIENT_ID and CLIENT_SECRET must exist in your environment / above this code
    main_vegetation(config_path=config_path, client_id=client_id, client_secret=client_secret)
