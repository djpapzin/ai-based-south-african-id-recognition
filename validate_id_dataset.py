"""
South African ID Book Dataset Validator
Run this script locally BEFORE uploading to Google Drive
"""

import os
import json
import cv2
import random
from pathlib import Path

def validate_dataset_structure(dataset_root):
    """Validate the directory structure and file organization"""
    print("\n🔍 Validating dataset structure...")
    
    required_structure = {
        'root': [
            ('train', 'dir'),
            ('val', 'dir')
        ],
        'train': [
            ('images', 'dir'),
            ('annotations.json', 'file')
        ],
        'val': [
            ('images', 'dir'),
            ('annotations.json', 'file')
        ]
    }

    errors = []
    
    # Check root directory
    if not os.path.exists(dataset_root):
        errors.append(f"Dataset root directory not found: {dataset_root}")
        return False, errors
    
    # Check main structure
    for item in required_structure['root']:
        path = os.path.join(dataset_root, item[0])
        if not os.path.exists(path):
            errors.append(f"Missing required {'directory' if item[1] == 'dir' else 'file'}: {path}")
    
    # Check train/val structure
    for split in ['train', 'val']:
        split_path = os.path.join(dataset_root, split)
        if os.path.exists(split_path):
            for item in required_structure[split]:
                path = os.path.join(split_path, item[0])
                if not os.path.exists(path):
                    errors.append(f"Missing {split} {'directory' if item[1] == 'dir' else 'file'}: {path}")
    
    if errors:
        print("❌ Structure validation failed:")
        for error in errors:
            print(f" - {error}")
        return False, errors
    
    print("✓ Directory structure valid")
    return True, []

def validate_coco_file(json_path, image_dir):
    """Validate a COCO format annotation file with detailed checks"""
    print(f"\n🔍 Validating COCO file: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {str(e)}")
        return False

    # Required top-level keys
    required_keys = ['images', 'annotations', 'categories']
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        return False

    # Image validation
    image_ids = {}
    for img in data['images']:
        if 'id' not in img:
            print("❌ Image missing 'id' field")
            return False
        if 'file_name' not in img:
            print("❌ Image missing 'file_name' field")
            return False
            
        # Check image exists
        img_path = os.path.join(image_dir, img['file_name'])
        if not os.path.exists(img_path):
            print(f"❌ Image file not found: {img_path}")
            return False
            
        image_ids[img['id']] = img['file_name']

    # Category validation
    categories = {}
    for cat in data['categories']:
        if 'id' not in cat:
            print("❌ Category missing 'id' field")
            return False
        if 'name' not in cat:
            print("❌ Category missing 'name' field")
            return False
        categories[cat['id']] = cat['name']

    # Annotation validation
    for ann in data['annotations']:
        # Required fields
        required_ann_fields = ['id', 'image_id', 'category_id', 'bbox']
        missing = [f for f in required_ann_fields if f not in ann]
        if missing:
            print(f"❌ Annotation missing fields: {missing}")
            return False
            
        # Check image_id exists
        if ann['image_id'] not in image_ids:
            print(f"❌ Annotation references invalid image_id: {ann['image_id']}")
            return False
            
        # Check category_id exists
        if ann['category_id'] not in categories:
            print(f"❌ Annotation references invalid category_id: {ann['category_id']}")
            return False
            
        # Validate bbox
        bbox = ann['bbox']
        if len(bbox) != 4:
            print(f"❌ Invalid bbox format. Expected 4 values, got {len(bbox)}")
            return False
            
        if any(not isinstance(v, (int, float)) for v in bbox):
            print("❌ Bbox contains non-numeric values")
            return False
            
        if any(v < 0 for v in bbox):
            print("❌ Bbox contains negative values")
            return False

    print(f"✓ COCO file valid (Images: {len(data['images'])}, Annotations: {len(data['annotations'])}, Categories: {len(data['categories'])})")
    return True

def visualize_annotations(dataset_root, split='train', num_samples=3):
    """Visualize annotations with detailed error reporting"""
    print(f"\n👀 Visualizing {split} annotations...")
    
    json_path = os.path.join(dataset_root, split, 'annotations.json')
    image_dir = os.path.join(dataset_root, split, 'images')
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load annotations: {str(e)}")
        return

    # Get random sample
    samples = random.sample(data['images'], min(num_samples, len(data['images'])))
    
    for img_info in samples:
        print("\n" + "="*50)
        print(f"Image: {img_info['file_name']}")
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue
            
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue
            
        # Get annotations for this image
        annotations = [a for a in data['annotations'] if a['image_id'] == img_info['id']]
        print(f"Found {len(annotations)} annotations")
        
        # Draw bounding boxes
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']
            category = next((c['name'] for c in data['categories'] if c['id'] == ann['category_id']), 'unknown')
            
            # Convert COCO bbox [x,y,w,h] to OpenCV [x1,y1,x2,y2]
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{category}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            print(f"Annotation {i+1}: {category} @ [{x},{y},{w},{h}]")
        
        # Display image
        cv2.imshow('Annotation Preview', cv2.resize(img, (800, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Set your local dataset path here
    dataset_root = "./merged_dataset"
    
    print("🛠️ South African ID Book Dataset Validation")
    print(f"Validating dataset at: {os.path.abspath(dataset_root)}")
    
    # 1. Validate directory structure
    structure_valid, errors = validate_dataset_structure(dataset_root)
    if not structure_valid:
        print("\n❌ Validation failed: Directory structure issues")
        return
    
    # 2. Validate COCO files
    coco_valid = True
    for split in ['train', 'val']:
        json_path = os.path.join(dataset_root, split, 'annotations.json')
        image_dir = os.path.join(dataset_root, split, 'images')
        
        if not validate_coco_file(json_path, image_dir):
            coco_valid = False
            print(f"❌ Validation failed for {split} annotations")
    
    if not coco_valid:
        return
    
    # 3. Check category consistency
    print("\n🔗 Checking category consistency between splits...")
    categories = {}
    for split in ['train', 'val']:
        json_path = os.path.join(dataset_root, split, 'annotations.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        split_cats = {c['id']: c['name'] for c in data['categories']}
        categories[split] = split_cats
    
    if categories['train'] != categories['val']:
        print("❌ Category IDs/names differ between train and val!")
        print("Train categories:", categories['train'])
        print("Val categories:", categories['val'])
        return
    else:
        print("✓ Category consistency verified")
    
    # 4. Visualize annotations
    print("\n🎨 Visualization checks (close windows to continue)")
    visualize_annotations(dataset_root, 'train')
    visualize_annotations(dataset_root, 'val')
    
    print("\n✅ All checks passed! Dataset is ready for upload.")

if __name__ == "__main__":
    main() 