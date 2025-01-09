import os
import requests
from pathlib import Path

def download_sam_weights():
    """Download the SAM model weights if they don't exist"""
    weights_path = Path("sam_vit_h_4b8939.pth")
    
    if weights_path.exists():
        print("SAM weights already exist")
        return
    
    print("Downloading SAM weights...")
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(weights_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
    
    print("Download complete!")

if __name__ == "__main__":
    download_sam_weights() 