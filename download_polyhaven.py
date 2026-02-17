import requests
import os

# --- Configuration ---
SAVE_DIR = "/projects/vig/Datasets/Polyhaven"
ASSET_TYPE = "hdris"  # Options: 'hdris', 'textures', 'models'
RESOLUTION = "4k"      # Options: '1k', '2k', '4k', '8k'
FILE_FORMAT = "exr"    # 'exr' or 'hdr' for HDRIs; 'png' or 'jpg' for textures

def download_polyhaven():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 1. Get the list of all assets
    api_url = f"https://api.polyhaven.com/assets?t={ASSET_TYPE}"
    headers = {'User-Agent': 'MyPolyDownloader/1.0'}
    
    print(f"Fetching list of {ASSET_TYPE}...")
    response = requests.get(api_url, headers=headers)
    assets = response.json()

    # 2. Loop through each asset and get the download link
    for asset_id in assets:
        print(f"Downloading {asset_id}...")
        
        # Get specific download info for this asset
        detail_url = f"https://api.polyhaven.com/files/{asset_id}"
        detail_res = requests.get(detail_url, headers=headers).json()

        try:
            # Navigate the JSON for the specific resolution/format
            # This path varies slightly between models and HDRIs
            if ASSET_TYPE == "hdris":
                download_url = detail_res['hdri'][RESOLUTION][FILE_FORMAT]['url']
                ext = FILE_FORMAT
            else:
                # For textures/models, it usually falls under 'blend' or 'zip'
                download_url = detail_res['blend']['url']
                ext = "blend"

            # 3. Download the file
            file_data = requests.get(download_url).content
            file_path = os.path.join(SAVE_DIR, f"{asset_id}.{ext}")
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
                
        except KeyError:
            print(f"Skipping {asset_id}: Resolution/Format not available.")

if __name__ == "__main__":
    download_polyhaven()