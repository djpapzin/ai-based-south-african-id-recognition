import json
import os
from collections import defaultdict
import numpy as np

def load_json(json_file):
    """Load JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {str(e)}")
        return None

def analyze_coco_annotations(data, dataset_name):
    """Analyze COCO format annotations for potential issues."""
    if not data:
        return
    
    print(f"\n=== {dataset_name} Analysis ===")
    
    # Basic counts
    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))
    num_categories = len(data.get('categories', []))
    
    print(f"\nBasic Statistics:")
    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_annotations}")
    print(f"Number of categories: {num_categories}")
    
    # Analyze categories
    categories = data.get('categories', [])
    category_ids = [cat['id'] for cat in categories]
    print("\nCategory Analysis:")
    print(f"Category IDs: {sorted(category_ids)}")
    print("Categories:")
    for cat in categories:
        print(f"  ID {cat['id']}: {cat.get('name', 'unnamed')}")
    
    if min(category_ids) != 1:
        print("WARNING: Category IDs don't start from 1")
    if len(category_ids) != max(category_ids):
        print("WARNING: Category IDs are not consecutive")
    
    # Analyze annotations
    print("\nAnnotation Analysis:")
    annotations = data.get('annotations', [])
    has_keypoints = False
    has_bbox = False
    keypoint_stats = defaultdict(int)
    bbox_stats = defaultdict(int)
    
    # Detailed keypoint analysis
    keypoint_fields = set()
    keypoint_formats = set()
    
    for ann in annotations:
        # Check all fields that might contain keypoint data
        for key in ann.keys():
            if 'point' in key.lower() or 'keypoint' in key.lower():
                keypoint_fields.add(key)
                keypoint_formats.add(str(type(ann[key])))
        
        # Standard COCO keypoints
        if 'keypoints' in ann:
            has_keypoints = True
            num_keypoints = len(ann['keypoints']) // 3  # COCO format: x,y,v for each keypoint
            keypoint_stats[num_keypoints] += 1
            
            # Check visibility values
            if num_keypoints > 0:
                visibilities = ann['keypoints'][2::3]  # Get every third value (visibility)
                for v in visibilities:
                    if v not in [0, 1, 2]:
                        print(f"WARNING: Invalid keypoint visibility value: {v}")
        
        # Check bounding boxes
        if 'bbox' in ann:
            has_bbox = True
            bbox = ann['bbox']
            if len(bbox) == 4:
                bbox_stats['valid'] += 1
            else:
                bbox_stats['invalid'] += 1
    
    if keypoint_fields:
        print("\nFound potential keypoint fields:")
        for field in keypoint_fields:
            print(f"  - {field} (type: {list(keypoint_formats)})")
            # Show sample of first annotation with this field
            for ann in annotations:
                if field in ann:
                    print(f"    Sample value: {ann[field]}")
                    break
    
    print(f"\nHas standard COCO keypoint annotations: {has_keypoints}")
    if has_keypoints:
        print("Keypoint distribution (points per annotation):")
        for num_points, count in sorted(keypoint_stats.items()):
            print(f"  {num_points} points: {count} annotations")
        
        # Check keypoint metadata
        if 'keypoints' in data.get('categories', [{}])[0]:
            print("\nKeypoint names defined in categories:")
            print(data['categories'][0].get('keypoints', []))
    
    print(f"\nHas bounding box annotations: {has_bbox}")
    if has_bbox:
        print("Bounding box statistics:")
        print(f"  Valid boxes (4 coordinates): {bbox_stats['valid']}")
        print(f"  Invalid boxes: {bbox_stats['invalid']}")
    
    # Sample annotation
    if annotations:
        print("\nSample annotation structure:")
        sample_ann = annotations[0]
        print(json.dumps(sample_ann, indent=2))

def analyze_label_studio_export(data):
    """Analyze Label Studio export format."""
    print("\n=== Label Studio Export Analysis ===")
    
    if not isinstance(data, list):
        print("Error: Expected list of annotations")
        return
    
    print(f"\nBasic Statistics:")
    print(f"Number of annotated images: {len(data)}")
    
    # Analyze annotation types
    annotation_types = defaultdict(int)
    keypoint_fields = set()
    bbox_fields = set()
    
    # Analyze first item in detail
    if data:
        print("\nAnnotation Structure Analysis:")
        first_item = data[0]
        print("Available fields:", list(first_item.keys()))
        
        if 'annotations' in first_item:
            print("\nAnnotations field contains:", list(first_item['annotations'][0].keys()) if first_item['annotations'] else "empty")
        
        # Look for keypoint-related fields
        for key, value in first_item.items():
            if isinstance(value, list) and value:
                if any('point' in str(x).lower() for x in value):
                    keypoint_fields.add(key)
                if any('bbox' in str(x).lower() or 'rectangle' in str(x).lower() for x in value):
                    bbox_fields.add(key)
    
    if keypoint_fields:
        print("\nFound potential keypoint fields:", list(keypoint_fields))
        # Show example of keypoint data
        for field in keypoint_fields:
            print(f"\nSample {field}:")
            print(json.dumps(data[0].get(field, [])[:2], indent=2))
    
    if bbox_fields:
        print("\nFound potential bounding box fields:", list(bbox_fields))
        # Show example of bbox data
        for field in bbox_fields:
            print(f"\nSample {field}:")
            print(json.dumps(data[0].get(field, [])[:2], indent=2))
    
    print("\nFull sample annotation:")
    print(json.dumps(data[0], indent=2))

def main():
    # Define paths
    result_min_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset\result_min.json"
    
    # Analyze result_min.json
    print("\n=== Analyzing result_min.json ===")
    data = load_json(result_min_path)
    if data:
        if 'images' in data and 'annotations' in data and 'categories' in data:
            analyze_coco_annotations(data, "COCO Format")
        elif isinstance(data, list):
            analyze_label_studio_export(data)
        else:
            print("Unsupported data format")
    else:
        print("Could not load result_min.json")

if __name__ == "__main__":
    main()
