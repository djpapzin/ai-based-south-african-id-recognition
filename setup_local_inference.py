import os
import shutil
from pathlib import Path

def setup_local_inference():
    """
    Create directory structure for local inference:
    project_root/
    ├── models/
    │   ├── model_final.pth (model weights)
    │   ├── model_cfg.yaml (model configuration)
    │   └── metadata.json (class names and metadata)
    ├── test_images/
    │   └── (put your test images here)
    ├── outputs/
    │   └── visualizations/
    └── detected_segments/
        └── (segmented results will be saved here)
    """
    
    # Define directory structure
    directories = {
        "models": "Store your model files (model_final.pth, model_cfg.yaml, metadata.json)",
        "test_images": "Place your test images here",
        "outputs/visualizations": "Visualization results will be saved here",
        "detected_segments": "Detected segments will be saved here",
        "logs": "Inference logs will be saved here"
    }
    
    # Create directories
    print("\nSetting up directory structure...")
    root_dir = Path.cwd()
    
    for dir_path, description in directories.items():
        full_path = root_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
        print(f"  → {description}")
    
    # Create README files in each directory
    for dir_path, description in directories.items():
        readme_path = root_dir / dir_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"# {dir_path}\n\n{description}\n")
    
    # Create model files checklist
    model_files_checklist = """# Required Model Files

Please copy the following files from your training output to the models/ directory:

1. [ ] model_final.pth (Model weights)
2. [ ] model_cfg.yaml (Model configuration)
3. [ ] metadata.json (Class names and metadata)

These files should be in your Google Drive at:
/content/drive/MyDrive/Kwantu/Machine Learning/model_output/
"""
    
    with open("models/README.md", "w") as f:
        f.write(model_files_checklist)
    
    # Create configuration file
    config = {
        "model": {
            "weights_path": "models/model_final.pth",
            "config_path": "models/model_cfg.yaml",
            "metadata_path": "models/metadata.json",
            "confidence_threshold": 0.5
        },
        "inference": {
            "use_gpu": False,
            "batch_size": 1,
            "save_visualizations": True
        },
        "paths": {
            "test_images": "test_images",
            "outputs": "outputs/visualizations",
            "segments": "detected_segments",
            "logs": "logs"
        }
    }
    
    import json
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\nNext steps:")
    print("1. Copy these files from your Google Drive to the models/ directory:")
    print("   - model_final.pth")
    print("   - model_cfg.yaml")
    print("   - metadata.json")
    print("2. Place your test images in the test_images/ directory")
    print("3. Run the inference script")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
.env
.venv/

# Model files
models/*.pth
models/*.yaml
models/*.json

# Output directories
outputs/
detected_segments/
logs/

# Config
config.json

# OS files
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("\nSetup complete! Directory structure is ready.")
    print(f"Project root: {root_dir}")

if __name__ == "__main__":
    setup_local_inference() 