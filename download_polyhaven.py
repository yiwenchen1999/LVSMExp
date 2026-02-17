import requests
import os

# --- Configuration ---
SAVE_DIR = "/projects/vig/Datasets/Polyhaven"
ASSET_TYPES = ["hdris", "textures", "models"]  # Download all in one pass
RESOLUTION = "8k"       # Options: '1k', '2k', '4k', '8k'
HDRI_FORMAT = "exr"     # 'exr' or 'hdr' for HDRIs
RESOLUTION_FALLBACK = ["1k", "2k", "4k", "8k", "16k"]  # Try in order if requested res not available

def get_best_resolution(available_resolutions):
    """Pick the best available resolution from fallback order."""
    for res in RESOLUTION_FALLBACK:
        if res in available_resolutions:
            return res
    return list(available_resolutions)[0] if available_resolutions else None

def download_hdri(asset_id, detail_res, save_dir, headers):
    """Download HDRI asset. Returns True on success."""
    hdri_data = detail_res.get("hdri")
    if not hdri_data:
        return False
    res = get_best_resolution(hdri_data.keys())
    if not res:
        return False
    formats = hdri_data[res]
    fmt = HDRI_FORMAT if HDRI_FORMAT in formats else list(formats.keys())[0]
    download_url = formats[fmt]["url"]
    ext = fmt
    file_path = os.path.join(save_dir, f"{asset_id}.{ext}")
    file_data = requests.get(download_url).content
    with open(file_path, "wb") as f:
        f.write(file_data)
    return True

def download_blend_asset(asset_id, detail_res, save_dir, headers):
    """Download texture or model as .blend. Returns True on success."""
    blend_data = detail_res.get("blend")
    if not blend_data:
        return False
    res = get_best_resolution(blend_data.keys())
    if not res:
        return False
    blend_inner = blend_data[res].get("blend")
    if not blend_inner:
        return False
    download_url = blend_inner["url"]
    file_path = os.path.join(save_dir, f"{asset_id}.blend")
    file_data = requests.get(download_url).content
    with open(file_path, "wb") as f:
        f.write(file_data)
    return True

def download_polyhaven():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    headers = {"User-Agent": "MyPolyDownloader/1.0"}

    for asset_type in ASSET_TYPES:
        type_dir = os.path.join(SAVE_DIR, asset_type)
        os.makedirs(type_dir, exist_ok=True)

        api_url = f"https://api.polyhaven.com/assets?t={asset_type}"
        print(f"Fetching list of {asset_type}...")
        response = requests.get(api_url, headers=headers)
        assets = response.json()

        for asset_id in assets:
            print(f"  [{asset_type}] {asset_id}...")
            detail_url = f"https://api.polyhaven.com/files/{asset_id}"
            detail_res = requests.get(detail_url, headers=headers).json()

            try:
                if asset_type == "hdris":
                    ok = download_hdri(asset_id, detail_res, type_dir, headers)
                elif asset_type in ("textures", "models"):
                    ok = download_blend_asset(asset_id, detail_res, type_dir, headers)
                else:
                    ok = False

                if not ok:
                    print(f"    Skipping {asset_id}: format not available.")
            except (KeyError, TypeError) as e:
                print(f"    Skipping {asset_id}: {e}.")

if __name__ == "__main__":
    download_polyhaven()
