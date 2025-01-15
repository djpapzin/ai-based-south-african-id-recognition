import argparse
import os
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def fix_image_paths(annotations_path: str) -> str:
    """Fix image paths in annotations file."""
    logger = logging.getLogger(__name__)
    logger.info("Fixing image paths in annotations file...")
    
    # Read annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Fix paths
    for img in data['images']:
        # Remove extra 'images' from path if present
        if 'images\\images\\' in img['file_name']:
            img['file_name'] = img['file_name'].replace('images\\images\\', '')
        elif 'images/images/' in img['file_name']:
            img['file_name'] = img['file_name'].replace('images/images/', '')
    
    # Save fixed annotations
    fixed_path = annotations_path.replace('.json', '_fixed.json')
    with open(fixed_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Fixed annotations saved to: {fixed_path}")
    return fixed_path

def check_requirements():
    """Check if all required packages are installed."""
    logger = logging.getLogger(__name__)
    logger.info("Checking requirements...")
    
    try:
        import torch
        import detectron2
        import cv2
        import numpy as np
        logger.info("All required packages are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.info("Please install required packages using:")
        logger.info("pip install torch detectron2 opencv-python numpy")
        return False

def run_pipeline(
    annotations_path: str,
    images_dir: str,
    output_dir: str,
    num_classes: int
):
    """
    Run the complete training pipeline.
    
    Args:
        annotations_path: Path to COCO format annotations JSON
        images_dir: Directory containing the images
        output_dir: Base directory for outputs
        num_classes: Number of object classes
    """
    logger = logging.getLogger(__name__)
    
    # Check requirements first
    if not check_requirements():
        return
    
    # Create output directories
    splits_dir = os.path.join(output_dir, "dataset_splits")
    training_dir = os.path.join(output_dir, "training_output")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(training_dir, exist_ok=True)
    
    # Fix image paths in annotations
    fixed_annotations = fix_image_paths(annotations_path)
    
    # 1. Validate annotations
    logger.info("Step 1: Validating annotations...")
    validation_cmd = (
        f"python test_coco_validation.py "
        f"--mode validate "
        f"--input {fixed_annotations} "
        f"--image-dir {images_dir}"
    )
    if os.system(validation_cmd) != 0:
        logger.error("Annotation validation failed!")
        return
    
    # 2. Split dataset
    logger.info("Step 2: Splitting dataset...")
    split_cmd = (
        f"python split_coco_dataset.py "
        f"--input {fixed_annotations} "
        f"--image-dir {images_dir} "
        f"--output-dir {splits_dir} "
        f"--train-split 0.9 "
        f"--val-split 0.1"
    )
    if os.system(split_cmd) != 0:
        logger.error("Dataset splitting failed!")
        return
    
    # 3. Start training
    logger.info("Step 3: Starting training...")
    train_cmd = (
        f"python train.py "
        f"--train-json {os.path.join(splits_dir, 'train.json')} "
        f"--train-image-dir {images_dir} "
        f"--val-json {os.path.join(splits_dir, 'val.json')} "
        f"--val-image-dir {images_dir} "
        f"--output-dir {training_dir} "
        f"--num-classes {num_classes}"
    )
    if os.system(train_cmd) != 0:
        logger.error("Training failed!")
        return
    
    logger.info("Pipeline completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Run the complete training pipeline')
    parser.add_argument('--annotations', required=True,
                      help='Path to COCO format annotations JSON')
    parser.add_argument('--images-dir', required=True,
                      help='Directory containing the images')
    parser.add_argument('--output-dir', default='pipeline_output',
                      help='Base directory for outputs')
    parser.add_argument('--num-classes', type=int, required=True,
                      help='Number of object classes')
    
    args = parser.parse_args()
    
    run_pipeline(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes
    )

if __name__ == "__main__":
    main() 