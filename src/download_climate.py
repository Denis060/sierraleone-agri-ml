"""
download_climate.py
-------------------
Acquire and cache the raw exogenous climate inputs.

  1. Sierra Leone national boundary (geoBoundaries ADM0, GeoJSON)
  2. CHIRPS v2.0 monthly Africa rainfall GeoTIFFs, 2000–2024 (300 files)
  3. NASA POWER monthly T2M + PRECTOTCORR at a 5-point national grid

Everything is cached under data/climate/ and re-runs skip files already present.

    python -m src.download_climate            # fetch all
    python -m src.download_climate --no-chirps # skip the heavy raster download

Author: Ibrahim Denis Fofanah — Pace University | RiseAfrica Foundation
"""

import os
import sys
import json
import time
import requests

from . import config


# ── 1. National boundary ──────────────────────────────────────────────────────
def download_boundary(verbose: bool = True) -> str:
    """Fetch Sierra Leone ADM0 GeoJSON via the geoBoundaries API; cache locally."""
    os.makedirs(config.CLIMATE_DIR, exist_ok=True)
    out = os.path.join(config.CLIMATE_DIR, 'sle_adm0.geojson')
    if os.path.exists(out):
        if verbose:
            print(f'[boundary] cached -> {out}')
        return out

    meta = requests.get(config.GEOBOUNDARIES_URL, timeout=60).json()
    gj_url = meta.get('gjDownloadURL') or meta.get('simplifiedGeometryGeoJSON')
    geojson = requests.get(gj_url, timeout=120).json()
    with open(out, 'w') as fh:
        json.dump(geojson, fh)
    if verbose:
        print(f'[boundary] downloaded {meta.get("boundaryName")} ADM0 -> {out}')
    return out


# ── 2. CHIRPS monthly rainfall rasters ────────────────────────────────────────
def download_chirps(verbose: bool = True) -> list:
    """Download every monthly CHIRPS Africa GeoTIFF for config.YEARS (cached)."""
    os.makedirs(config.CHIRPS_DIR, exist_ok=True)
    paths, downloaded, skipped, failed = [], 0, 0, []

    for year in config.YEARS:
        for month in range(1, 13):
            fname = f'chirps-v2.0.{year}.{month:02d}.tif.gz'
            url   = config.CHIRPS_BASE_URL + fname
            dest  = os.path.join(config.CHIRPS_DIR, fname)
            paths.append(dest)

            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                skipped += 1
                continue

            for attempt in range(3):
                try:
                    with requests.get(url, stream=True, timeout=180) as r:
                        r.raise_for_status()
                        tmp = dest + '.part'
                        with open(tmp, 'wb') as fh:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                fh.write(chunk)
                        os.replace(tmp, dest)
                    downloaded += 1
                    break
                except Exception as exc:
                    if attempt == 2:
                        failed.append(fname)
                        print(f'[chirps] FAILED {fname}: {exc}')
                    else:
                        time.sleep(2 * (attempt + 1))

            if verbose and (downloaded + skipped) % 24 == 0:
                print(f'[chirps] progress: {downloaded} new, {skipped} cached '
                      f'({downloaded + skipped}/{len(config.YEARS) * 12})')

    print(f'[chirps] done: {downloaded} downloaded, {skipped} cached, '
          f'{len(failed)} failed')
    if failed:
        print(f'[chirps] failures: {failed}')
    return paths


# ── 3. NASA POWER monthly temperature + precip (5-point grid) ─────────────────
def download_nasa_power(verbose: bool = True) -> str:
    """
    Fetch monthly T2M and PRECTOTCORR for each grid point and cache the raw
    per-point JSON responses. Returns the directory holding them.
    """
    raw_dir = os.path.join(config.CLIMATE_DIR, 'nasa_power_raw')
    os.makedirs(raw_dir, exist_ok=True)
    y0, y1 = config.YEARS[0], config.YEARS[-1]

    for i, (lat, lon) in enumerate(config.NASA_POINTS):
        out = os.path.join(raw_dir, f'point_{i}_{lat}_{lon}.json')
        if os.path.exists(out):
            if verbose:
                print(f'[power] cached point {i} ({lat},{lon})')
            continue
        params = {
            'parameters': 'T2M,PRECTOTCORR', 'community': 'AG',
            'longitude': lon, 'latitude': lat,
            'start': y0, 'end': y1, 'format': 'JSON',
        }
        resp = requests.get(config.NASA_POWER_URL, params=params, timeout=120)
        resp.raise_for_status()
        with open(out, 'w') as fh:
            json.dump(resp.json(), fh)
        if verbose:
            print(f'[power] downloaded point {i} ({lat},{lon})')
        time.sleep(1)   # be polite to the API
    return raw_dir


def main():
    do_chirps = '--no-chirps' not in sys.argv
    download_boundary()
    download_nasa_power()
    if do_chirps:
        download_chirps()
    else:
        print('[chirps] skipped (--no-chirps)')


if __name__ == '__main__':
    main()
