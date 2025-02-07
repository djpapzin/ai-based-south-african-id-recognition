import subprocess
import sys
import os

def install_detectron2():
    """Install prebuilt Detectron2 wheel for Python 3.9."""
    print("Installing Detectron2 prebuilt wheel...")
    
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "-f", "https://download.pytorch.org/whl/torch_stable.html"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "numpy"])
    
    # Download and install prebuilt wheel for Python 3.9
    wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/detectron2-0.6%2Bcpu-cp39-cp39-win_amd64.whl"
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
    
    print("Detectron2 installation complete!")

if __name__ == "__main__":
    install_detectron2()
