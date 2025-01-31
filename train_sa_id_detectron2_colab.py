"""South African ID Book Detection Training with Detectron2

This script trains a Detectron2 model for detecting South African ID books using a pre-split dataset.
The script is designed to work in both Google Colab (with GPU) and local environments (CPU).
"""

# Install dependencies (uncomment these lines when running in Colab)
# !pip install torch torchvision
# !pip install 'git+https://github.com/facebookresearch/detectron2.git'

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
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer

# Setup logger
setup_logger()

def check_gpu():
    """Check GPU availability and print device information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return torch.cuda.is_available()

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

def verify_dataset(json_path, images_dir):
    """Verify dataset integrity."""
    print(f"\nVerifying dataset at {json_path}")
    
    # Check JSON file
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                coco_data = json.load(f)
                print(f"✓ JSON file is valid")
                print(f"  - Number of images: {len(coco_data['images'])}")
                print(f"  - Number of annotations: {len(coco_data['annotations'])}")
                print(f"  - Categories: {[cat['name'] for cat in coco_data['categories']]}")
            except json.JSONDecodeError:
                print("✗ Error: Invalid JSON file")
                return False
    else:
        print("✗ Error: JSON file not found")
        return False
    
    # Check images directory
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ Found {len(image_files)} images")
        
        # Verify a random image
        if image_files:
            test_image = os.path.join(images_dir, random.choice(image_files))
            img = cv2.imread(test_image)
            if img is not None:
                print(f"✓ Successfully loaded test image")
            else:
                print("✗ Error: Failed to load test image")
                return False
    else:
        print("✗ Error: Images directory not found")
        return False
    
    return True

def visualize_prediction(predictor, image_path):
    """Visualize model prediction on an image."""
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    # Convert BGR to RGB for visualization
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Create visualizer
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("sa_id_val"),
                   scale=0.8)
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(15, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.show()

def main():
    # Check GPU availability
    use_gpu = check_gpu()
    print(f"\nUsing {'GPU' if use_gpu else 'CPU'} for training")
    
    # Set paths for your existing dataset structure
    GDRIVE_PATH = "/content/drive/MyDrive/Kwantu/Machine Learning"
    DATASET_PATH = os.path.join(GDRIVE_PATH, "merged_dataset")
    TRAIN_PATH = os.path.join(DATASET_PATH, "train")
    VAL_PATH = os.path.join(DATASET_PATH, "val")
    OUTPUT_DIR = os.path.join(GDRIVE_PATH, "model_output")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Verify training dataset
    train_json = os.path.join(TRAIN_PATH, "annotations.json")
    train_images = os.path.join(TRAIN_PATH, "images")
    if not verify_dataset(train_json, train_images):
        print("Training dataset verification failed")
        return
    
    # Verify validation dataset
    val_json = os.path.join(VAL_PATH, "annotations.json")
    val_images = os.path.join(VAL_PATH, "images")
    if not verify_dataset(val_json, val_images):
        print("Validation dataset verification failed")
        return
    
    # Register datasets
    print("\nRegistering datasets...")
    register_coco_instances(
        "sa_id_train",
        {},
        train_json,
        train_images
    )
    register_coco_instances(
        "sa_id_val",
        {},
        val_json,
        val_images
    )
    
    # Setup configuration
    print("\nSetting up model configuration...")
    cfg = setup_cfg(
        train_dataset_name="sa_id_train",
        val_dataset_name="sa_id_val",
        num_classes=15,  # Updated to match the actual number of classes
        output_dir=OUTPUT_DIR,
        use_gpu=use_gpu
    )
    
    # Create trainer
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    print("\nStarting training...")
    trainer.train()
    
    # Create predictor for visualization
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    
    # Visualize predictions on validation images
    print("\nVisualizing predictions on validation images...")
    val_images_list = os.listdir(val_images)
    for i in range(min(3, len(val_images_list))):
        image_path = os.path.join(val_images, val_images_list[i])
        print(f"\nPredictions for {val_images_list[i]}:")
        visualize_prediction(predictor, image_path)

if __name__ == "__main__":
    main()
