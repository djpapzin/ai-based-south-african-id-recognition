from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import detection_utils
import detectron2.data.transforms as T
import logging
import json
import cv2
import os
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dataloader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def validate_coco_dataset(
    json_file: str,
    image_dir: str
) -> Tuple[bool, Dict]:
    """
    Validate COCO dataset format and contents.
    
    Args:
        json_file: Path to COCO JSON file
        image_dir: Directory containing the images
        
    Returns:
        Tuple of (is_valid, stats_dict)
    """
    try:
        # Load JSON file
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        # Initialize statistics
        stats = {
            'total_images': 0,
            'valid_images': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'categories': set(),
            'image_sizes': set(),
            'annotations_per_image': [],
            'category_counts': {},
            'missing_images': [],
            'invalid_annotations': []
        }
        
        # Validate required fields
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                logging.error(f"Missing required field: {field}")
                return False, stats
        
        # Validate categories
        for cat in coco_data['categories']:
            if not all(k in cat for k in ['id', 'name']):
                logging.error(f"Invalid category format: {cat}")
                return False, stats
            stats['categories'].add(cat['name'])
            stats['category_counts'][cat['name']] = 0
        
        # Create category ID to name mapping
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Validate images
        image_ids = set()
        stats['total_images'] = len(coco_data['images'])
        
        for img in coco_data['images']:
            # Validate image entry
            if not all(k in img for k in ['id', 'file_name', 'height', 'width']):
                logging.error(f"Invalid image format: {img}")
                continue
            
            # Check image file existence
            img_path = os.path.join(image_dir, img['file_name'])
            if not os.path.exists(img_path):
                logging.error(f"Image file not found: {img_path}")
                stats['missing_images'].append(img['file_name'])
                continue
            
            # Verify image dimensions
            actual_img = cv2.imread(img_path)
            if actual_img is None:
                logging.error(f"Could not read image: {img_path}")
                continue
                
            actual_height, actual_width = actual_img.shape[:2]
            if actual_width != img['width'] or actual_height != img['height']:
                logging.error(f"Image dimensions mismatch for {img_path}. "
                            f"JSON: {img['width']}x{img['height']}, "
                            f"Actual: {actual_width}x{actual_height}")
                continue
            
            image_ids.add(img['id'])
            stats['valid_images'] += 1
            stats['image_sizes'].add((img['width'], img['height']))
        
        # Validate annotations
        stats['total_annotations'] = len(coco_data['annotations'])
        annotations_by_image = {}
        
        for ann in coco_data['annotations']:
            # Validate annotation entry
            if not all(k in ann for k in ['id', 'image_id', 'category_id', 'bbox']):
                logging.error(f"Invalid annotation format: {ann}")
                stats['invalid_annotations'].append(ann)
                continue
            
            # Validate image reference
            if ann['image_id'] not in image_ids:
                logging.error(f"Annotation references non-existent image_id: {ann['image_id']}")
                continue
            
            # Validate category reference
            if ann['category_id'] not in cat_id_to_name:
                logging.error(f"Invalid category_id in annotation: {ann}")
                continue
            
            # Validate bbox format
            if len(ann['bbox']) != 4:
                logging.error(f"Invalid bbox format: {ann['bbox']}")
                continue
            
            # Update statistics
            cat_name = cat_id_to_name[ann['category_id']]
            stats['category_counts'][cat_name] += 1
            stats['valid_annotations'] += 1
            
            # Group annotations by image
            if ann['image_id'] not in annotations_by_image:
                annotations_by_image[ann['image_id']] = []
            annotations_by_image[ann['image_id']].append(ann)
        
        # Calculate annotations per image statistics
        stats['annotations_per_image'] = [
            len(anns) for anns in annotations_by_image.values()
        ]
        
        # Log statistics
        logging.info("Dataset Statistics:")
        logging.info(f"Images: {stats['valid_images']}/{stats['total_images']}")
        logging.info(f"Annotations: {stats['valid_annotations']}/{stats['total_annotations']}")
        logging.info(f"Categories: {len(stats['categories'])}")
        logging.info("Category distribution:")
        for cat, count in stats['category_counts'].items():
            logging.info(f"  {cat}: {count}")
        logging.info(f"Image sizes: {stats['image_sizes']}")
        if stats['annotations_per_image']:
            logging.info(f"Annotations per image: min={min(stats['annotations_per_image'])}, "
                        f"max={max(stats['annotations_per_image'])}, "
                        f"avg={sum(stats['annotations_per_image'])/len(stats['annotations_per_image']):.1f}")
        
        return True, stats
        
    except Exception as e:
        logging.error(f"Validation failed with error: {str(e)}")
        return False, stats

def register_datasets(
    train_json: str,
    train_image_dir: str,
    val_json: Optional[str] = None,
    val_image_dir: Optional[str] = None,
    dataset_name: str = "my_dataset"
) -> bool:
    """
    Register training and validation datasets with Detectron2.
    
    Args:
        train_json: Path to training set COCO JSON
        train_image_dir: Path to training images directory
        val_json: Path to validation set COCO JSON (optional)
        val_image_dir: Path to validation images directory (optional)
        dataset_name: Base name for the dataset
        
    Returns:
        bool: True if registration succeeds
    """
    try:
        # Validate and register training set
        logging.info("Validating training dataset...")
        is_valid, train_stats = validate_coco_dataset(train_json, train_image_dir)
        if not is_valid:
            logging.error("Training dataset validation failed")
            return False
        
        register_coco_instances(
            f"{dataset_name}_train",
            {},
            train_json,
            train_image_dir
        )
        
        # Validate and register validation set if provided
        if val_json and val_image_dir:
            logging.info("Validating validation dataset...")
            is_valid, val_stats = validate_coco_dataset(val_json, val_image_dir)
            if not is_valid:
                logging.error("Validation dataset validation failed")
                return False
            
            register_coco_instances(
                f"{dataset_name}_val",
                {},
                val_json,
                val_image_dir
            )
        
        return True
        
    except Exception as e:
        logging.error(f"Dataset registration failed with error: {str(e)}")
        return False

def visualize_dataset_samples(
    dataset_name: str,
    output_dir: str,
    num_samples: int = 5
) -> None:
    """
    Visualize random samples from the dataset to verify loading.
    
    Args:
        dataset_name: Name of the registered dataset
        output_dir: Directory to save visualization images
        num_samples: Number of samples to visualize
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dataset and metadata
        dataset_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        
        # Randomly sample images
        samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
        
        for i, d in enumerate(samples):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
            vis = visualizer.draw_dataset_dict(d)
            
            # Save visualization
            output_path = os.path.join(output_dir, f"sample_{i}.png")
            cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
            
            # Log bounding box information
            logging.info(f"Sample {i} - {d['file_name']}:")
            for ann in d["annotations"]:
                bbox = ann["bbox"]
                cat_id = ann["category_id"]
                logging.info(f"  Category {cat_id}, bbox: {bbox}")
        
        logging.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logging.error(f"Visualization failed with error: {str(e)}")

def get_data_loader(
    cfg,
    is_train: bool = True,
    mapper: Optional[callable] = None
):
    """
    Build a data loader for training or validation.
    
    Args:
        cfg: Detectron2 config object
        is_train: Whether this is a training data loader
        mapper: Optional custom mapper for data augmentation
        
    Returns:
        data_loader: Detectron2 data loader
    """
    if is_train:
        dataset_name = cfg.DATASETS.TRAIN[0]
    else:
        dataset_name = cfg.DATASETS.TEST[0]
    
    # Use custom mapper if provided, otherwise use default
    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=is_train)
    
    return build_detection_train_loader(
        cfg,
        dataset_name,
        mapper=mapper
    ) if is_train else build_detection_test_loader(
        cfg,
        dataset_name,
        mapper=mapper
    )

def main():
    """Example usage of the data loading utilities."""
    # Register datasets
    success = register_datasets(
        train_json="path/to/train.json",
        train_image_dir="path/to/train/images",
        val_json="path/to/val.json",
        val_image_dir="path/to/val/images",
        dataset_name="my_dataset"
    )
    
    if not success:
        logging.error("Failed to register datasets")
        return
    
    # Visualize samples
    visualize_dataset_samples(
        "my_dataset_train",
        "visualization_output",
        num_samples=5
    )

if __name__ == "__main__":
    main() 