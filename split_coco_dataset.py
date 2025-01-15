import json
import os
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
from coco_utils import validate_coco_format

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dataset_split_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def split_dataset(
    coco_file: str,
    image_dir: str,
    output_dir: str,
    train_split: float = 0.9,
    val_split: float = 0.1,
    test_split: float = 0.0,
    random_seed: int = 42
) -> bool:
    """
    Split a COCO format dataset into train and validation sets (and optionally test set).
    
    Args:
        coco_file (str): Path to the input COCO JSON file
        image_dir (str): Directory containing the images
        output_dir (str): Directory to save the split dataset files
        train_split (float): Proportion of data for training (default: 0.9)
        val_split (float): Proportion of data for validation (default: 0.1)
        test_split (float): Proportion of data for testing (default: 0.0)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        bool: True if splitting succeeds, False otherwise
    """
    try:
        # Validate input COCO file
        logging.info("Validating input COCO file...")
        if not validate_coco_format(coco_file, image_dir):
            logging.error("Input COCO file validation failed")
            return False
            
        # Load COCO JSON
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
            
        # Verify split proportions
        if abs(train_split + val_split + test_split - 1.0) > 1e-10:
            logging.error("Split proportions must sum to 1.0")
            return False
            
        # Set random seed for reproducibility
        random.seed(random_seed)
        
        # Get all image IDs and shuffle them
        image_ids = [img['id'] for img in coco_data['images']]
        random.shuffle(image_ids)
        
        # Calculate split sizes
        total_images = len(image_ids)
        train_size = int(total_images * train_split)
        val_size = int(total_images * val_split)
        
        # Split image IDs
        train_ids = set(image_ids[:train_size])
        val_ids = set(image_ids[train_size:train_size + val_size])
        test_ids = set(image_ids[train_size + val_size:]) if test_split > 0 else set()
        
        # Create split datasets
        splits = {
            'train': train_ids,
            'val': val_ids
        }
        if test_split > 0:
            splits['test'] = test_ids
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each split
        for split_name, split_ids in splits.items():
            # Initialize split dataset with categories
            split_data = {
                'categories': coco_data['categories'],
                'images': [],
                'annotations': []
            }
            
            # Add images for this split
            split_data['images'] = [
                img for img in coco_data['images']
                if img['id'] in split_ids
            ]
            
            # Add corresponding annotations
            split_data['annotations'] = [
                ann for ann in coco_data['annotations']
                if ann['image_id'] in split_ids
            ]
            
            # Save split dataset
            output_file = os.path.join(output_dir, f'{split_name}.json')
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            # Validate split dataset
            logging.info(f"Validating {split_name} split...")
            if not validate_coco_format(output_file, image_dir):
                logging.error(f"Validation failed for {split_name} split")
                return False
                
            # Log split statistics
            logging.info(f"{split_name} split statistics:")
            logging.info(f"  Images: {len(split_data['images'])}")
            logging.info(f"  Annotations: {len(split_data['annotations'])}")
            logging.info(f"  Categories: {len(split_data['categories'])}")
            
            # Log category distribution
            category_counts = {}
            for ann in split_data['annotations']:
                cat_id = ann['category_id']
                category_name = next(cat['name'] for cat in split_data['categories'] if cat['id'] == cat_id)
                category_counts[category_name] = category_counts.get(category_name, 0) + 1
            
            logging.info("  Category distribution:")
            for cat_name, count in sorted(category_counts.items()):
                logging.info(f"    {cat_name}: {count} annotations")
            
        logging.info("Dataset splitting completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Dataset splitting failed with error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Split COCO dataset into train/val sets')
    parser.add_argument('--input', required=True,
                      help='Input COCO JSON file path')
    parser.add_argument('--image-dir', required=True,
                      help='Directory containing the images')
    parser.add_argument('--output-dir', required=True,
                      help='Directory to save the split datasets')
    parser.add_argument('--train-split', type=float, default=0.9,
                      help='Proportion of data for training (default: 0.9)')
    parser.add_argument('--val-split', type=float, default=0.1,
                      help='Proportion of data for validation (default: 0.1)')
    parser.add_argument('--test-split', type=float, default=0.0,
                      help='Proportion of data for testing (default: 0.0)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()
    
    # Verify total split proportions
    total_split = args.train_split + args.val_split + args.test_split
    if abs(total_split - 1.0) > 1e-10:
        parser.error("Split proportions must sum to 1.0")
    
    success = split_dataset(
        args.input,
        args.image_dir,
        args.output_dir,
        args.train_split,
        args.val_split,
        args.test_split,
        args.seed
    )
    
    if success:
        splits = ["train", "val"]
        if args.test_split > 0:
            splits.append("test")
        print(f"✅ Dataset successfully split into {'/'.join(splits)} sets!")
        print(f"Output files saved in: {args.output_dir}")
        print("Check the log file for detailed statistics of each split")
    else:
        print("❌ Dataset splitting failed. Check the log file for details.")

if __name__ == "__main__":
    main() 