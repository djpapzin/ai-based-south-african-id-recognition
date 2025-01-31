import json
import os
import cv2
from pathlib import Path

def fix_dataset(json_path, images_dir):
    print(f"\nFixing dataset: {os.path.basename(json_path)}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Original stats:")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in data['categories']]}")
    
    # Fix image dimensions and create image_id lookup
    image_lookup = {}
    valid_images = []
    for img in data['images']:
        img_path = os.path.join(images_dir, img['file_name'])
        if os.path.exists(img_path):
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    h, w = image.shape[:2]
                    img['height'] = h
                    img['width'] = w
                    image_lookup[img['id']] = img
                    valid_images.append(img)
            except Exception as e:
                print(f"Error processing image {img['file_name']}: {e}")
    
    # Update images list
    data['images'] = valid_images
    
    # Fix annotations
    valid_annotations = []
    next_ann_id = 1
    
    for ann in data['annotations']:
        # Skip if image was invalid
        if ann.get('image_id') not in image_lookup:
            continue
            
        # Ensure all required fields are present
        if not all(key in ann for key in ['bbox', 'category_id', 'image_id']):
            continue
            
        # Validate bbox
        if not isinstance(ann['bbox'], list) or len(ann['bbox']) != 4:
            continue
            
        if not all(isinstance(x, (int, float)) for x in ann['bbox']):
            continue
            
        # Ensure bbox is within image bounds
        img = image_lookup[ann['image_id']]
        x, y, w, h = ann['bbox']
        if x < 0 or y < 0 or x + w > img['width'] or y + h > img['height']:
            continue
            
        # Add required fields
        ann['id'] = next_ann_id
        next_ann_id += 1
        ann['area'] = float(w * h)
        ann['iscrowd'] = 0
        ann['bbox_mode'] = 0  # XYWH_ABS
        
        valid_annotations.append(ann)
    
    # Update annotations
    data['annotations'] = valid_annotations
    
    # Ensure category IDs are sequential starting from 1
    category_map = {cat['id']: idx + 1 for idx, cat in enumerate(data['categories'])}
    for cat in data['categories']:
        cat['id'] = category_map[cat['id']]
    for ann in data['annotations']:
        ann['category_id'] = category_map[ann['category_id']]
    
    # Save fixed dataset
    output_path = str(Path(json_path).with_name('fixed_' + Path(json_path).name))
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"\nFixed stats:")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Output saved to: {output_path}")

# Fix both datasets
base_dir = os.path.dirname(__file__)
abenathi_json = os.path.join(base_dir, 'abenathi_object_detection_dataset', 'detectron2_coco.json')
dj_json = os.path.join(base_dir, 'dj_object_detection_dataset', 'detectron2_coco.json')

abenathi_images = os.path.join(base_dir, 'abenathi_object_detection_dataset', 'images')
dj_images = os.path.join(base_dir, 'dj_object_detection_dataset', 'images')

fix_dataset(abenathi_json, abenathi_images)
fix_dataset(dj_json, dj_images)
