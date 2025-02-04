import json
from pathlib import Path
import numpy as np

def verify_and_update_keypoints():
    """Verify and update keypoint annotations to match Detectron2 requirements"""
    project_dir = Path(__file__).parent
    train_ann = project_dir / "merged_dataset" / "train" / "annotations.json"
    val_ann = project_dir / "merged_dataset" / "val" / "annotations.json"
    
    # Define keypoint names for corners
    KEYPOINT_NAMES = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    
    def process_annotations(ann_file):
        print(f"\nProcessing {ann_file}...")
        with open(ann_file) as f:
            data = json.load(f)
        
        # Add keypoint information to categories
        for cat in data['categories']:
            if cat['name'] == 'id_document':
                cat['keypoints'] = KEYPOINT_NAMES
                cat['skeleton'] = [[0, 1], [1, 2], [2, 3], [3, 0]]  # Connect corners
        
        # Verify keypoint format
        keypoint_stats = {
            'total_annotations': len(data['annotations']),
            'with_keypoints': 0,
            'valid_format': 0,
            'invalid_format': 0
        }
        
        for ann in data['annotations']:
            if 'keypoints' in ann:
                keypoint_stats['with_keypoints'] += 1
                
                # Verify keypoint format
                keypoints = ann['keypoints']
                if len(keypoints) == 12:  # 4 keypoints * 3 values (x,y,v)
                    keypoint_stats['valid_format'] += 1
                else:
                    keypoint_stats['invalid_format'] += 1
                    print(f"Warning: Invalid keypoint format in annotation {ann['id']}")
        
        print("\nKeypoint Statistics:")
        print(f"Total annotations: {keypoint_stats['total_annotations']}")
        print(f"Annotations with keypoints: {keypoint_stats['with_keypoints']}")
        print(f"Valid keypoint format: {keypoint_stats['valid_format']}")
        print(f"Invalid keypoint format: {keypoint_stats['invalid_format']}")
        
        # Save updated annotations
        output_file = ann_file.parent / f"annotations_with_keypoints.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved updated annotations to: {output_file}")
        
        return keypoint_stats
    
    # Process both splits
    train_stats = process_annotations(train_ann)
    val_stats = process_annotations(val_ann)
    
    # Create Detectron2 dataset registration code
    print("\nGenerating Detectron2 dataset registration code...")
    
    registration_code = '''
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Register datasets
register_coco_instances(
    "sa_id_train",
    {},
    "merged_dataset/train/annotations_with_keypoints.json",
    "merged_dataset/train/images"
)

register_coco_instances(
    "sa_id_val",
    {},
    "merged_dataset/val/annotations_with_keypoints.json",
    "merged_dataset/val/images"
)

# Add keypoint metadata
keypoint_names = ''' + str(KEYPOINT_NAMES) + '''
keypoint_flip_map = []  # No flipping needed for document corners
keypoint_connection_rules = [
    ("top_left", "top_right", (102, 204, 255)),
    ("top_right", "bottom_right", (102, 204, 255)),
    ("bottom_right", "bottom_left", (102, 204, 255)),
    ("bottom_left", "top_left", (102, 204, 255)),
]

# Update metadata for both splits
for split in ["sa_id_train", "sa_id_val"]:
    meta = MetadataCatalog.get(split)
    meta.keypoint_names = keypoint_names
    meta.keypoint_flip_map = keypoint_flip_map
    meta.keypoint_connection_rules = keypoint_connection_rules
'''
    
    # Save registration code
    reg_file = project_dir / "register_sa_id_dataset.py"
    with open(reg_file, 'w') as f:
        f.write(registration_code)
    print(f"\nSaved dataset registration code to: {reg_file}")

if __name__ == "__main__":
    print("Starting keypoint verification and update...")
    verify_and_update_keypoints()
