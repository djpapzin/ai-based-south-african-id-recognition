import json
from pathlib import Path
import numpy as np

def validate_coco_annotations(annotation_path):
    """Validate COCO format annotations with detailed error checking"""
    try:
        with open(annotation_path) as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {str(e)}")
        return

    # Check top-level structure
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            print(f"Missing required top-level key: {key}")

    # Create lookup dictionaries
    image_ids = {img['id']: img for img in coco_data['images']}
    category_ids = {cat['id']: cat for cat in coco_data['categories']}

    # New validation checks
    def _validate_categories(annotations):
        expected_categories = 12
        unique_categories = set(ann['category_id'] for ann in annotations['annotations'])
        
        if len(unique_categories) != expected_categories:
            raise ValueError(f'Category mismatch: Expected {expected_categories} categories, found {len(unique_categories)}')

    def _validate_bboxes(annotations):
        for i, ann in enumerate(annotations['annotations']):
            bbox = ann['bbox']
            if len(bbox) != 4:
                raise ValueError(f'Invalid bbox at index {i}: Expected 4 values, got {len(bbox)}')
            
            if any(not (0 <= x <= 1) for x in bbox):
                raise ValueError(f'Invalid bbox values at index {i}: Values must be normalized between 0-1')

    _validate_categories(coco_data)
    _validate_bboxes(coco_data)

    # Validate annotations
    for i, ann in enumerate(coco_data['annotations']):
        # Check required annotation fields
        required_ann_fields = ['id', 'image_id', 'category_id', 'bbox']
        for field in required_ann_fields:
            if field not in ann:
                print(f"Annotation {i} missing required field: {field}")

        # Verify image exists
        if ann['image_id'] not in image_ids:
            print(f"Annotation {i} references missing image ID: {ann['image_id']}")

        # Verify category exists
        if ann['category_id'] not in category_ids:
            print(f"Annotation {i} has invalid category ID: {ann['category_id']}")

        # Validate bbox coordinates
        bbox = ann['bbox']
        if len(bbox) != 4:
            print(f"Annotation {i} has invalid bbox dimensions: {bbox}")
        elif any(not isinstance(x, (int, float)) for x in bbox):
            print(f"Annotation {i} has non-numeric bbox values: {bbox}")
        elif any(x < 0 for x in bbox):
            print(f"Annotation {i} has negative bbox values: {bbox}")

    print(f"Validation complete for {annotation_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python annotation_validator.py <dataset_root>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    for split in ['train', 'val']:
        ann_path = dataset_root / split / 'annotations.json'
        if not ann_path.exists():
            print(f"[!] Missing {split} annotations: {ann_path}")
            continue
        print(f"[VALIDATING] {ann_path}")
        validate_coco_annotations(ann_path)
