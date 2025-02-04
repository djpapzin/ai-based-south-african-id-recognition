import os
import cv2
import json
from pathlib import Path

def fix_dimensions(dataset_dir):
    """Fix image dimensions in annotations file."""
    annotations_file = os.path.join(dataset_dir, "annotations.json")
    images_dir = os.path.join(dataset_dir, "images")
    
    # Load annotations
    print(f"Loading annotations from {annotations_file}")
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Update dimensions for each image
    fixed_count = 0
    for img in data['images']:
        file_name = os.path.basename(img['file_name'].replace("\\", "/"))
        image_path = os.path.join(images_dir, file_name)
        
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                if img['width'] != width or img['height'] != height:
                    print(f"Fixing dimensions for {file_name}: {width}x{height}")
                    img['width'] = width
                    img['height'] = height
                    fixed_count += 1
    
    if fixed_count > 0:
        # Create backup
        backup_file = annotations_file + '.backup'
        print(f"Creating backup at {backup_file}")
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save updated annotations
        print(f"Saving {fixed_count} dimension fixes to {annotations_file}")
        with open(annotations_file, 'w') as f:
            json.dump(data, f, indent=2)
        print("✓ Save complete")
    else:
        print("No dimension fixes needed")

def main():
    dataset_dir = "merged_dataset"
    
    # Fix both train and val sets
    for split in ['train', 'val']:
        split_dir = os.path.join(dataset_dir, split)
        print(f"\nFixing {split} set dimensions...")
        fix_dimensions(split_dir)

if __name__ == "__main__":
    main() 