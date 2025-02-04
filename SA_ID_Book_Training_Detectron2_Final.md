# South African ID Book Detection Training with Detectron2 (Colab Version)

## 1. Install Dependencies

```python
import subprocess
import sys

def run_pip_install(package):
    print(f"\nInstalling {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    print(f"Successfully installed {package}")

# Install required packages
print("Starting dependency installation...")
packages = [
    'torch',
    'torchvision',
    'git+https://github.com/facebookresearch/detectron2.git'
]

for package in packages:
    run_pip_install(package)
print("\nAll dependencies installed successfully!")
```

## 2. Import Libraries and Setup Environment

```python
print("Importing required libraries...")

import os
import cv2
import json
import random
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer

# Setup logger
setup_logger()

# Print environment information
print("\nEnvironment Information:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"OpenCV version: {cv2.__version__}")
print("\nAll libraries imported successfully!")
```

## 3. Mount Drive and Setup Project Directory

```python
from google.colab import drive

def mount_google_drive():
    """Mount Google Drive and ensure connection."""
    try:
        drive.mount('/content/drive')
            print("✓ Google Drive mounted successfully!")
            return True
    except Exception as e:
        print(f"✗ Error mounting Google Drive: {str(e)}")
        return False

# Mount drive and set project directory
if mount_google_drive():
    DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"
    TRAIN_JSON = os.path.join(DRIVE_ROOT, "merged_dataset/train/annotations_keypoints.json")
    VAL_JSON = os.path.join(DRIVE_ROOT, "merged_dataset/val/annotations_keypoints.json")
    TRAIN_IMGS = os.path.join(DRIVE_ROOT, "merged_dataset/train/images")
    VAL_IMGS = os.path.join(DRIVE_ROOT, "merged_dataset/val/images")
    OUTPUT_DIR = os.path.join(DRIVE_ROOT, "model_output")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
else:
    raise RuntimeError("Failed to mount Google Drive. Please check your connection and try again.")

# Check GPU availability
use_gpu = torch.cuda.is_available()
print(f"\nUsing {'GPU' if use_gpu else 'CPU'} for training")
if use_gpu:
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Add after mounting Google Drive
def verify_colab_dataset():
    """Verify dataset is properly uploaded to Google Drive"""
    print("\nVerifying dataset in Google Drive:")
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        
        # Check annotation file
        ann_path = os.path.join(DRIVE_ROOT, f"merged_dataset/{split}/annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Annotations: {len(data['images'])} images, {len(data['annotations'])} annotations")
        else:
            print(f"✗ Missing annotations file: {ann_path}")
        
        # Check images directory
        img_dir = os.path.join(DRIVE_ROOT, f"merged_dataset/{split}/images")
        if os.path.exists(img_dir):
            images = list(Path(img_dir).glob("*.jpg"))
            print(f"✓ Images directory: {len(images)} images")
        else:
            print(f"✗ Missing images directory: {img_dir}")

# Run verification after mounting
verify_colab_dataset()
```

## 4. Register and Verify Dataset

```python
class CocoTrainer(DefaultTrainer):
    """Custom trainer class with COCO evaluator."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir, use_gpu=True):
    """Setup Detectron2 configuration."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Dataset configuration
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2

    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True

    # Training configuration
    if use_gpu:
        cfg.SOLVER.IMS_PER_BATCH = 2
    else:
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.DEVICE = "cpu"

    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg
```

## 5. Configure and Train Model

```python
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg(train_dataset="sa_id_train", val_dataset="sa_id_val"):
    """Set up model configuration for keypoint detection."""
    cfg = get_cfg()
    
    # Load Keypoint R-CNN base config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset config
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    
    # Initialize from Keypoint R-CNN pretrained model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    
    # Keypoint head config
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 4  # 4 corners
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0
    
    # Model config
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(categories)  # Number of object classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    # Training config
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []  # No learning rate decay
    
    # Input config
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Output config
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.TENSORBOARD.ENABLED = True
    cfg.TENSORBOARD.LOG_DIR = LOG_DIR
    
    return cfg

# Set up configuration
cfg = setup_cfg()

print("\nModel Configuration:")
print(f"Number of keypoints: {cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS}")
print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"Training iterations: {cfg.SOLVER.MAX_ITER}")
print(f"Learning rate: {cfg.SOLVER.BASE_LR}")

# Initialize trainer
trainer = CustomTrainer(cfg)

# Start training
print("\nStarting training...")
trainer.resume_or_load(resume=False)
trainer.train()
```

## 6. Save and Export Model

```python
# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "model_final.pth")
print(f"\nSaving final model to: {final_model_path}")

# Export model config
cfg_path = os.path.join(OUTPUT_DIR, "model_config.yaml")
with open(cfg_path, 'w') as f:
    f.write(cfg.dump())
print(f"Saved model config to: {cfg_path}")

# Function to visualize predictions with keypoints
def visualize_predictions(predictor, image_path):
    """Visualize model predictions including keypoints on an image."""
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    # Create visualizer
    v = Visualizer(im[:, :, ::-1],
                  metadata=MetadataCatalog.get("sa_id_val"),
                  scale=0.5)
    
    # Draw instance predictions
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display using matplotlib
    plt.figure(figsize=(15, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.title("Model Predictions with Keypoints")
    plt.show()

# Test the model on a validation image
from detectron2.engine import DefaultPredictor

predictor = DefaultPredictor(cfg)
val_images = os.listdir(VAL_IMGS)
if val_images:
    test_image = os.path.join(VAL_IMGS, val_images[0])
    print(f"\nTesting model on: {test_image}")
    visualize_predictions(predictor, test_image)