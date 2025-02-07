import subprocess
import sys
import os
import winreg
import urllib.request
import zipfile

def download_file(url, filename):
    """Download a file from a URL."""
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)

def install_python39():
    """Download and install Python 3.9."""
    python_installer = "python-3.9.13-amd64.exe"
    if not os.path.exists(python_installer):
        download_file(
            "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe",
            python_installer
        )
    
    print("Installing Python 3.9...")
    subprocess.check_call([python_installer, "/quiet", "InstallAllUsers=0", "PrependPath=1"])

def get_python39_path():
    """Get Python 3.9 installation path from registry."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Python\PythonCore\3.9\InstallPath") as key:
            return winreg.QueryValue(key, None)
    except:
        return None

def setup_detectron2():
    """Set up Detectron2 with Python 3.9."""
    # Check if Python 3.9 is installed
    python39_path = get_python39_path()
    if not python39_path:
        install_python39()
        python39_path = get_python39_path()
    
    if not python39_path:
        print("Error: Could not find Python 3.9 installation")
        return
    
    python_exe = os.path.join(python39_path, "python.exe")
    pip_exe = os.path.join(python39_path, "Scripts", "pip.exe")
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.check_call([pip_exe, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_exe, "install", "torch==2.0.1+cpu", "torchvision==0.15.2+cpu", "-f", "https://download.pytorch.org/whl/torch_stable.html"])
    subprocess.check_call([pip_exe, "install", "opencv-python", "numpy"])
    
    # Install Detectron2 prebuilt wheel
    print("Installing Detectron2...")
    wheel_url = "https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/detectron2-0.6%2Bcpu-cp39-cp39-win_amd64.whl"
    subprocess.check_call([pip_exe, "install", wheel_url])
    
    print("Setup complete! You can now use Detectron2 with Python 3.9")
    print(f"Python 3.9 path: {python_exe}")

if __name__ == "__main__":
    setup_detectron2()
