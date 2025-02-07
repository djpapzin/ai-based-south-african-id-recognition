import subprocess
import sys
import os

def install_detectron2():
    """Install Detectron2 from source on Windows."""
    print("Installing Detectron2 from source...")
    
    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pycocotools-windows"])
    
    # Clone and install Detectron2
    if not os.path.exists("detectron2"):
        subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/detectron2.git"])
    
    # Install Detectron2
    os.chdir("detectron2")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    os.chdir("..")
    
    print("Detectron2 installation complete!")

if __name__ == "__main__":
    install_detectron2()
