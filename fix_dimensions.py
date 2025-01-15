import json
import os
import cv2
from pathlib import Path

def fix_image_dimensions(coco_file: str, image_dir: str, output_file: str) -> None:
    """Fix image dimensions in COCO annotations by reading actual image sizes."""
    
    print(f"Loading COCO annotations from {coco_file}")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    print("Fixing image dimensions...")
    for img in coco_data['images']:
        img_path = os.path.join(image_dir, img['file_name'])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        actual_img = cv2.imread(img_path)
        if actual_img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
        
        actual_height, actual_width = actual_img.shape[:2]
        if actual_width != img['width'] or actual_height != img['height']:
            print(f"Fixing dimensions for {img['file_name']}")
            print(f"  Old: {img['width']}x{img['height']}")
            print(f"  New: {actual_width}x{actual_height}")
            img['width'] = actual_width
            img['height'] = actual_height
    
    print(f"Saving fixed annotations to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    coco_file = "annotated_images/result_fixed.json"
    image_dir = "annotated_images/images"
    output_file = "annotated_images/result_fixed_dimensions.json"
    
    fix_image_dimensions(coco_file, image_dir, output_file) 