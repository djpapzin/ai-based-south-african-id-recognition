import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, check=True)
    return process.returncode == 0

def install_detectron2():
    # Install PyTorch CPU
    print("Installing PyTorch CPU...")
    run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

    # Install dependencies
    print("Installing dependencies...")
    run_command("pip install cython numpy")
    run_command("pip install opencv-python")
    run_command("pip install pycocotools")

    # Clone Detectron2
    print("Cloning Detectron2...")
    if os.path.exists("detectron2"):
        print("Detectron2 directory already exists, removing it...")
        run_command("rmdir /s /q detectron2")
    
    run_command("git clone https://github.com/facebookresearch/detectron2.git")
    
    # Install Detectron2
    print("Installing Detectron2...")
    os.chdir("detectron2")
    run_command("python setup.py install")
    os.chdir("..")

    print("Installation complete! Testing import...")
    try:
        import detectron2
        print("Detectron2 installed successfully!")
    except ImportError as e:
        print(f"Error importing detectron2: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_detectron2() 