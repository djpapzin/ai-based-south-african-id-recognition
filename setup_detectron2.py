import os
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    # Install build tools
    run_command("pip install ninja")
    
    # Clone Detectron2
    if os.path.exists("detectron2"):
        print("Removing existing detectron2 directory...")
        run_command("rmdir /s /q detectron2")
    
    run_command("git clone https://github.com/facebookresearch/detectron2.git")
    os.chdir("detectron2")
    
    # Install in develop mode
    run_command("pip install -e .")
    
    # Verify installation
    os.chdir("..")
    run_command('python -c "import detectron2; print(\'Detectron2 installed successfully!\')"')

if __name__ == "__main__":
    main() 