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
    TRAIN_JSON = os.path.join(DRIVE_ROOT, "merged_dataset/train/annotations.json")
    VAL_JSON = os.path.join(DRIVE_ROOT, "merged_dataset/val/annotations.json")
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
# Clear any existing registrations
if "sa_id_train" in DatasetCatalog:
    DatasetCatalog.remove("sa_id_train")
if "sa_id_val" in DatasetCatalog:
    DatasetCatalog.remove("sa_id_val")
if "sa_id_train" in MetadataCatalog:
    MetadataCatalog.remove("sa_id_train")
if "sa_id_val" in MetadataCatalog:
    MetadataCatalog.remove("sa_id_val")

# Define categories and metadata
categories = [
    {"id": 0, "name": "bottom_left_corner", "supercategory": "corner"},
    {"id": 1, "name": "bottom_right_corner", "supercategory": "corner"},
    {"id": 2, "name": "citizenship_status", "supercategory": "field"},
    {"id": 3, "name": "country_of_birth", "supercategory": "field"},
    {"id": 4, "name": "date_of_birth", "supercategory": "field"},
    {"id": 5, "name": "face", "supercategory": "field"},
    {"id": 6, "name": "id_document", "supercategory": "document"},
    {"id": 7, "name": "id_number", "supercategory": "field"},
    {"id": 8, "name": "names", "supercategory": "field"},
    {"id": 9, "name": "nationality", "supercategory": "field"},
    {"id": 10, "name": "sex", "supercategory": "field"},
    {"id": 11, "name": "signature", "supercategory": "field"},
    {"id": 12, "name": "surname", "supercategory": "field"},
    {"id": 13, "name": "top_left_corner", "supercategory": "corner"},
    {"id": 14, "name": "top_right_corner", "supercategory": "corner"}
]

# Create metadata dict
metadata_dict = {
    "thing_classes": [cat["name"] for cat in categories],
    "thing_dataset_id_to_contiguous_id": {cat["id"]: i for i, cat in enumerate(categories)},
    "keypoint_names": ["top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner"],
    "keypoint_flip_map": [],  # No flipping for document corners
    "keypoint_connection_rules": [
        ("top_left_corner", "top_right_corner", (102, 204, 255)),  # top edge
        ("top_right_corner", "bottom_right_corner", (102, 204, 255)),  # right edge
        ("bottom_right_corner", "bottom_left_corner", (102, 204, 255)),  # bottom edge
        ("bottom_left_corner", "top_left_corner", (102, 204, 255))  # left edge
    ]
}

# Print paths for verification
print("\nVerifying annotation paths:")
print(f"Train annotations: {TRAIN_JSON}")
print(f"Train images: {TRAIN_IMGS}")
print(f"Val annotations: {VAL_JSON}")
print(f"Val images: {VAL_IMGS}")

# Verify files exist
for path in [TRAIN_JSON, VAL_JSON, TRAIN_IMGS, VAL_IMGS]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    print(f"✓ Found: {path}")

# Register datasets
print("\nRegistering datasets...")
register_coco_instances(
    "sa_id_train",
    metadata_dict,
    TRAIN_JSON,
    TRAIN_IMGS
)
register_coco_instances(
    "sa_id_val",
    metadata_dict,
    VAL_JSON,
    VAL_IMGS
)

# Verify registration and visualize sample with keypoints
def visualize_sample_with_keypoints(dataset_name="sa_id_train", num_samples=2):
    """Visualize sample images with keypoints from the dataset."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    for d in random.sample(dataset_dicts, num_samples):
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"Could not read image: {d['file_name']}")
            continue
            
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        
        # Draw all annotations
        vis = visualizer.draw_dataset_dict(d)
        
        # Check for corner annotations
        corner_anns = [ann for ann in d["annotations"] 
                      if ann["category_id"] in [0, 1, 13, 14]]  # Corner category IDs
        
        if corner_anns:
            print(f"\nFound {len(corner_anns)} corner annotations in {d['file_name']}")
            for ann in corner_anns:
                corner_type = metadata.thing_classes[ann["category_id"]]
                bbox = ann["bbox"]
                print(f"Corner: {corner_type}")
                print(f"Position: x={bbox[0]:.1f}, y={bbox[1]:.1f}")
                
                # Draw corner points with labels
                x, y = int(bbox[0]), int(bbox[1])
                cv2.circle(vis.output.get_image(), (x, y), 5, (0, 255, 0), -1)
                cv2.putText(vis.output.get_image(), corner_type, 
                          (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
            
            # Draw connections between corners if we have all four
            corner_points = {}
            for ann in corner_anns:
                corner_type = metadata.thing_classes[ann["category_id"]]
                bbox = ann["bbox"]
                corner_points[corner_type] = (int(bbox[0]), int(bbox[1]))
            
            if len(corner_points) == 4:
                corners_ordered = ["top_left_corner", "top_right_corner", 
                                 "bottom_right_corner", "bottom_left_corner"]
                for i in range(4):
                    pt1 = corner_points[corners_ordered[i]]
                    pt2 = corner_points[corners_ordered[(i + 1) % 4]]
                    cv2.line(vis.output.get_image(), pt1, pt2, (102, 204, 255), 2)
        else:
            print(f"\nNo corner annotations found in {d['file_name']}")
        
        # Display using matplotlib
        plt.figure(figsize=(15, 10))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.title(f"Sample from {dataset_name}")
        plt.show()

# Verify registration
print("\nRegistered datasets:")
print(f"Training: {len(DatasetCatalog.get('sa_id_train'))} images")
print(f"Validation: {len(DatasetCatalog.get('sa_id_val'))} images")

print("\nVisualizing training samples with keypoints:")
visualize_sample_with_keypoints("sa_id_train")
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
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11  # Number of object classes
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