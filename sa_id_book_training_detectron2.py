# -*- coding: utf-8 -*-
"""SA ID Book Training Detectron2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e4rZi5vZwLsxVcRMDyHNanRnQ5KHHV9O

# Install dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install torch torchvision
# !pip install 'git+https://github.com/facebookresearch/detectron2.git'

"""# Import required libraries and check GPU"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

# Check if running in Colab
try:
    from google.colab import drive
    IN_COLAB = True
    print("Running in Colab environment")
    drive.mount('/content/drive')
except ImportError:
    IN_COLAB = False
    print("Running in local environment")

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg(train_json, val_json, num_classes, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("id_card_train",)
    cfg.DATASETS.TEST = ("id_card_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = DEVICE
    return cfg

def train_model(train_json, val_json, train_images_dir, val_images_dir, output_dir, num_classes):
    print("Setting up datasets...")
    register_coco_instances("id_card_train", {}, train_json, train_images_dir)
    register_coco_instances("id_card_val", {}, val_json, val_images_dir)

    print("Configuring model...")
    cfg = setup_cfg(train_json, val_json, num_classes, output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("Starting training...")
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def setup_predictor(model_path, num_classes=7, confidence_threshold=0.7):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = DEVICE
    return DefaultPredictor(cfg)

def process_image(image_path, predictor, save_dir="detected_regions"):
    """Process a single image with detection and visualization"""
    print(f"\nProcessing: {Path(image_path).name}")
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Run inference
    outputs = predictor(im)
    
    # Save detected regions
    class_names = ["Date of Birth", "Face Photo", "ID Number", "Names", "Sex", "Surname", "Type of ID"]
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (128,128,0)]
    
    # Create output directory
    image_name = Path(image_path).stem
    regions_dir = os.path.join(save_dir, image_name)
    os.makedirs(regions_dir, exist_ok=True)
    
    # Get predictions
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    
    # Save individual regions
    for box, class_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.astype(int)
        region = im[y1:y2, x1:x2]
        if region.size == 0:
            continue
            
        filename = f"{class_names[class_id]}_{score:.2f}.jpg"
        save_path = os.path.join(regions_dir, filename)
        cv2.imwrite(save_path, region)
        
    # Visualize results
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    for box, class_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[class_id]
        
        # Draw box
        cv2.rectangle(im_rgb, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        label = f"{class_names[class_id]}: {score:.2f}"
        font_scale = 1.0
        thickness = 3
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        label_y = y1 - 10 if y1 - label_size[1] - 10 >= 0 else y1 + label_size[1] + 10
        padding = 8
        
        # Draw label background
        cv2.rectangle(im_rgb,
                     (x1, label_y - label_size[1] - padding),
                     (x1 + label_size[0] + padding, label_y + padding),
                     (255, 255, 255),
                     -1)
        
        # Draw label border
        cv2.rectangle(im_rgb,
                     (x1, label_y - label_size[1] - padding),
                     (x1 + label_size[0] + padding, label_y + padding),
                     color,
                     2)
        
        # Add text
        cv2.putText(im_rgb, label,
                   (x1 + padding//2, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale,
                   (0, 0, 0),
                   thickness)
    
    # Display result
    plt.figure(figsize=(15, 10))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.title(Path(image_path).name, fontsize=12, pad=20)
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train or run inference on ID card dataset')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True,
                      help='Whether to train a model or run inference')
    
    # Training arguments
    parser.add_argument('--train-json', help='Path to training JSON file')
    parser.add_argument('--val-json', help='Path to validation JSON file')
    parser.add_argument('--train-images-dir', help='Path to training images directory')
    parser.add_argument('--val-images-dir', help='Path to validation images directory')
    parser.add_argument('--output-dir', help='Path to output directory')
    
    # Inference arguments
    parser.add_argument('--model-path', help='Path to trained model for inference')
    parser.add_argument('--image-dir', help='Directory containing images for inference')
    parser.add_argument('--num-images', type=int, help='Number of images to process (for inference)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not all([args.train_json, args.val_json, args.train_images_dir, 
                   args.val_images_dir, args.output_dir]):
            parser.error("Training mode requires all training-related arguments")
        train_model(args.train_json, args.val_json, args.train_images_dir, 
                   args.val_images_dir, args.output_dir, num_classes=7)
    else:
        if not all([args.model_path, args.image_dir]):
            parser.error("Inference mode requires model-path and image-dir arguments")
        
        predictor = setup_predictor(args.model_path)
        images = [f for f in os.listdir(args.image_dir) 
                 if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        
        if args.num_images:
            images = random.sample(images, min(args.num_images, len(images)))
        
        for image_name in images:
            image_path = os.path.join(args.image_dir, image_name)
            process_image(image_path, predictor)

if __name__ == "__main__":
    main()