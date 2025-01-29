import json
import os
import shutil
from sklearn.model_selection import train_test_split
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_actual_filename(image_dir, coco_filename):
    """
    Convert COCO filename (with UUID) to actual filename.
    Example: 'd8d826bf-0000000000000_A_A.jpg' -> '0000000000000_A_A.jpg'
    """
    # Extract ID and suffix from COCO filename
    match = re.search(r'[a-f0-9]+-(\d+_[A-Z](?:_[A-Z])?\.(?:jpg|jpeg))', coco_filename)
    if not match:
        return None
    
    actual_filename = match.group(1)
    actual_path = os.path.join(image_dir, actual_filename)
    
    # Check if file exists
    if os.path.exists(actual_path):
        return actual_filename
    return None

def prepare_dataset(coco_file, image_dir, output_dir, train_split=0.8):
    """
    Prepare dataset for Colab by splitting into train and validation sets.
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Load COCO annotations
    logger.info(f"Loading annotations from {coco_file}")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get valid images (those we can find on disk)
    valid_images = []
    for img in coco_data['images']:
        actual_filename = get_actual_filename(os.path.join(image_dir, 'images'), img['file_name'].replace('images\\', ''))
        if actual_filename:
            img['file_name'] = os.path.join('images', actual_filename)
            valid_images.append(img)
        else:
            logger.warning(f"Could not find matching file for {img['file_name']}")
    
    logger.info(f"Found {len(valid_images)} valid images out of {len(coco_data['images'])} total images")
    
    # Get image IDs and split
    image_ids = [img['id'] for img in valid_images]
    train_ids, val_ids = train_test_split(image_ids, train_size=train_split, random_state=42)
    
    # Create new COCO annotation files for train and val
    train_coco = {'images': [], 'annotations': [], 'categories': coco_data['categories']}
    val_coco = {'images': [], 'annotations': [], 'categories': coco_data['categories']}
    
    # Process images and annotations
    id_to_split = {id_: 'train' for id_ in train_ids}
    id_to_split.update({id_: 'val' for id_ in val_ids})
    
    # Copy images and split annotations
    for img in valid_images:
        img_id = img['id']
        split = id_to_split[img_id]
        
        # Get source and destination paths
        src_path = os.path.join(image_dir, img['file_name'])
        if split == 'train':
            dst_dir = train_dir
            train_coco['images'].append(img)
        else:
            dst_dir = val_dir
            val_coco['images'].append(img)
            
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.join(dst_dir, 'images'), exist_ok=True)
        
        # Copy image file
        dst_path = os.path.join(dst_dir, img['file_name'])
        try:
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {os.path.basename(src_path)} to {split} set")
        except FileNotFoundError:
            logger.error(f"Could not find image file: {src_path}")
            continue
    
    # Split annotations
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in train_ids:
            train_coco['annotations'].append(ann)
        elif img_id in val_ids:
            val_coco['annotations'].append(ann)
    
    # Save split COCO files
    with open(os.path.join(train_dir, 'train.json'), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(val_dir, 'val.json'), 'w') as f:
        json.dump(val_coco, f)
    
    # Print statistics
    logger.info(f"Dataset split complete:")
    logger.info(f"Training set: {len(train_ids)} images, {len(train_coco['annotations'])} annotations")
    logger.info(f"Validation set: {len(val_ids)} images, {len(val_coco['annotations'])} annotations")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare dataset for Colab')
    parser.add_argument('--coco-file', required=True, help='Path to COCO JSON file')
    parser.add_argument('--image-dir', required=True, help='Path to image directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for split dataset')
    parser.add_argument('--train-split', type=float, default=0.8, help='Proportion of training set')
    
    args = parser.parse_args()
    prepare_dataset(args.coco_file, args.image_dir, args.output_dir, args.train_split) 