import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Define the unified categories
UNIFIED_CATEGORIES = [
    {"id": 0, "name": "id_document", "type": "bbox"},
    {"id": 1, "name": "surname", "type": "bbox"},
    {"id": 2, "name": "names", "type": "bbox"},
    {"id": 3, "name": "sex", "type": "bbox"},  # New ID only
    {"id": 4, "name": "nationality", "type": "bbox"},  # New ID only
    {"id": 5, "name": "id_number", "type": "bbox"},
    {"id": 6, "name": "date_of_birth", "type": "bbox"},
    {"id": 7, "name": "country_of_birth", "type": "bbox"},
    {"id": 8, "name": "citizenship_status", "type": "bbox"},
    {"id": 9, "name": "face", "type": "bbox"},
    {"id": 10, "name": "signature", "type": "bbox"},  # New ID only
    {"id": 11, "name": "top_left_corner", "type": "keypoint"},
    {"id": 12, "name": "top_right_corner", "type": "keypoint"},
    {"id": 13, "name": "bottom_left_corner", "type": "keypoint"},
    {"id": 14, "name": "bottom_right_corner", "type": "keypoint"}
]

def create_category_maps():
    """Create mapping dictionaries for old and new ID formats"""
    # Map from old category IDs to new unified IDs
    old_id_map = {
        'id_document': 0,
        'surname': 1,
        'names': 2,
        'id_number': 5,
        'date_of_birth': 6,
        'country_of_birth': 7,
        'citizenship_status': 8,
        'face': 9,
        'top_left_corner': 11,
        'top_right_corner': 12,
        'bottom_left_corner': 13,
        'bottom_right_corner': 14
    }
    
    # Map for new ID categories
    new_id_map = {
        'id_document': 0,
        'surname': 1,
        'names': 2,
        'sex': 3,
        'nationality': 4,
        'id_number': 5,
        'date_of_birth': 6,
        'country_of_birth': 7,
        'citizenship_status': 8,
        'face': 9,
        'signature': 10,
        'top_left_corner': 11,
        'top_right_corner': 12,
        'bottom_left_corner': 13,
        'bottom_right_corner': 14
    }
    
    return old_id_map, new_id_map

def preprocess_image(image_path, output_path, target_size=(800, 800)):
    """Preprocess image to standardize size and quality"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Failed to read image"
        
        # Preserve aspect ratio
        h, w = img.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create blank canvas
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Center image on canvas
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Save processed image
        cv2.imwrite(str(output_path), canvas)
        return True, (scale, x_offset, y_offset)
    except Exception as e:
        return False, str(e)

def adjust_annotations(annotations, scale, x_offset, y_offset):
    """Adjust annotations based on image preprocessing"""
    for ann in annotations:
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            ann['bbox'] = [
                x * scale + x_offset,
                y * scale + y_offset,
                w * scale,
                h * scale
            ]
        if 'keypoints' in ann:
            for i in range(0, len(ann['keypoints']), 3):
                ann['keypoints'][i] = ann['keypoints'][i] * scale + x_offset
                ann['keypoints'][i+1] = ann['keypoints'][i+1] * scale + y_offset
    return annotations

def validate_dataset(dataset_path):
    """Validate the dataset for common issues"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    issues = []
    
    # Check basic structure
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in data:
            issues.append(f"Missing required key: {key}")
            return issues
    
    # Check image IDs
    image_ids = set(img['id'] for img in data['images'])
    if len(image_ids) != len(data['images']):
        issues.append("Duplicate image IDs found")
    
    # Check image files exist
    image_dir = Path(dataset_path).parent / 'images'
    for img in data['images']:
        if not (image_dir / img['file_name']).exists():
            issues.append(f"Missing image file: {img['file_name']}")
    
    # Check annotations
    for ann in data['annotations']:
        # Check image reference
        if ann['image_id'] not in image_ids:
            issues.append(f"Invalid image_id in annotation {ann['id']}")
        
        # Check category
        if ann['category_id'] >= len(data['categories']):
            issues.append(f"Invalid category_id in annotation {ann['id']}")
        
        # Check bbox format for rectangle annotations
        if 'bbox' in ann:
            if len(ann['bbox']) != 4:
                issues.append(f"Invalid bbox format in annotation {ann['id']}")
            if any(not isinstance(x, (int, float)) for x in ann['bbox']):
                issues.append(f"Invalid bbox values in annotation {ann['id']}")
        
        # Check keypoint format
        if 'keypoints' in ann:
            if len(ann['keypoints']) % 3 != 0:
                issues.append(f"Invalid keypoint format in annotation {ann['id']}")
            if any(not isinstance(x, (int, float)) for x in ann['keypoints']):
                issues.append(f"Invalid keypoint values in annotation {ann['id']}")
    
    return issues

def fix_annotations(dataset_path, is_new_id_format, output_dir):
    """Fix annotations for a single dataset"""
    print(f"Processing {'new' if is_new_id_format else 'old'} ID format dataset...")
    
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    old_id_map, new_id_map = create_category_maps()
    category_map = new_id_map if is_new_id_format else old_id_map
    
    # Process images and annotations
    valid_annotations = []
    processed_images = []
    
    for idx, img in enumerate(tqdm(data['images'], desc="Processing images")):
        # Get image path
        img_path = Path(dataset_path).parent / 'images' / img['file_name']
        if not img_path.exists():
            print(f"Warning: Image not found: {img['file_name']}")
            continue
        
        # Preprocess image
        output_img_path = output_dir / 'images' / img['file_name'].replace('.jpg', '.jpeg')
        success, result = preprocess_image(img_path, output_img_path)
        
        if not success:
            print(f"Warning: Failed to process image {img['file_name']}: {result}")
            continue
        
        scale, x_offset, y_offset = result
        
        # Update image entry
        img['id'] = len(processed_images)
        img['file_name'] = img['file_name'].replace('.jpg', '.jpeg')
        img['width'] = 800
        img['height'] = 800
        processed_images.append(img)
        
        # Process annotations for this image
        img_annotations = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]
        img_annotations = adjust_annotations(img_annotations, scale, x_offset, y_offset)
        
        for ann in img_annotations:
            ann['id'] = len(valid_annotations)
            ann['image_id'] = img['id']
            valid_annotations.append(ann)
    
    # Update dataset
    output_data = {
        'images': processed_images,
        'annotations': valid_annotations,
        'categories': UNIFIED_CATEGORIES
    }
    
    # Save processed dataset
    output_json = output_dir / 'annotations.json'
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return len(processed_images), len(valid_annotations)

def merge_datasets(old_id_dir, new_id_dir, output_dir):
    """Merge and fix both datasets"""
    print("Starting dataset merge process...")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    
    # Process both datasets
    old_images, old_annotations = fix_annotations(
        old_id_dir,
        False,
        output_dir / 'old_processed'
    )
    
    new_images, new_annotations = fix_annotations(
        new_id_dir,
        True,
        output_dir / 'new_processed'
    )
    
    # Load processed datasets
    with open(output_dir / 'old_processed' / 'annotations.json', 'r') as f:
        old_data = json.load(f)
    with open(output_dir / 'new_processed' / 'annotations.json', 'r') as f:
        new_data = json.load(f)
    
    # Merge datasets
    merged_data = {
        'images': old_data['images'] + new_data['images'],
        'annotations': old_data['annotations'] + new_data['annotations'],
        'categories': UNIFIED_CATEGORIES
    }
    
    # Reindex everything
    for idx, img in enumerate(merged_data['images']):
        old_id = img['id']
        img['id'] = idx
        # Update corresponding annotations
        for ann in merged_data['annotations']:
            if ann['image_id'] == old_id:
                ann['image_id'] = idx
    
    for idx, ann in enumerate(merged_data['annotations']):
        ann['id'] = idx
    
    # Copy all images to final location
    print("Copying processed images to final location...")
    for source_dir in ['old_processed', 'new_processed']:
        for img_file in (output_dir / source_dir / 'images').glob('*'):
            shutil.copy2(img_file, output_dir / 'images' / img_file.name)
    
    # Save merged dataset
    with open(output_dir / 'annotations.json', 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # Cleanup temporary directories
    shutil.rmtree(output_dir / 'old_processed')
    shutil.rmtree(output_dir / 'new_processed')
    
    print("\nValidating merged dataset...")
    issues = validate_dataset(output_dir / 'annotations.json')
    if issues:
        print("\nValidation issues found:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nNo validation issues found!")
    
    return len(merged_data['images']), len(merged_data['annotations'])

def main():
    """Main execution function"""
    # Define paths
    old_id_dir = Path("abenathi_object_detection_dataset/detectron2_coco.json")
    new_id_dir = Path("dj_object_detection_dataset/detectron2_coco.json")
    output_dir = Path("merged_dataset")
    
    print("Starting dataset processing...")
    
    try:
        total_images, total_annotations = merge_datasets(old_id_dir, new_id_dir, output_dir)
        
        print(f"\nProcessing complete!")
        print(f"Total images: {total_images}")
        print(f"Total annotations: {total_annotations}")
        print(f"\nOutput saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 