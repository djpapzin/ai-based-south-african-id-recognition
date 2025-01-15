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
        logging.FileHandler(f'data_preparation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
        # Remove any existing path and just keep the filename
        filename = os.path.basename(img['file_name'])
        img['file_name'] = filename
    
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
        import cv2
        import numpy as np
        logger.info("All required packages are installed.")
        return True
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.info("Please install required packages using:")
        logger.info("pip install opencv-python numpy")
        return False

def prepare_data(
    annotations_path: str,
    images_dir: str,
    output_dir: str
):
    """
    Prepare dataset by validating and splitting the data.
    
    Args:
        annotations_path: Path to COCO format annotations JSON
        images_dir: Directory containing the images
        output_dir: Base directory for outputs
    """
    logger = logging.getLogger(__name__)
    
    # Check requirements first
    if not check_requirements():
        return
    
    # Create output directories
    splits_dir = os.path.join(output_dir, "dataset_splits")
    os.makedirs(splits_dir, exist_ok=True)
    
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
    
    logger.info(f"""
Data preparation completed successfully!
Outputs:
- Fixed annotations: {fixed_annotations}
- Train/Val splits: {splits_dir}
    - train.json
    - val.json

You can now proceed with training using these files.
""")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--annotations', required=True,
                      help='Path to COCO format annotations JSON')
    parser.add_argument('--images-dir', required=True,
                      help='Directory containing the images')
    parser.add_argument('--output-dir', default='dataset_preparation',
                      help='Base directory for outputs')
    
    args = parser.parse_args()
    
    prepare_data(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 