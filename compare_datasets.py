import json
import os
from collections import defaultdict

def load_coco_json(json_path):
    """Load and return the contents of a COCO format JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return None

def analyze_dataset(data, dataset_name):
    """Analyze the contents of a COCO dataset."""
    if not data:
        print(f"No data available for {dataset_name}")
        return None
    
    analysis = {
        'num_images': len(data.get('images', [])),
        'num_annotations': len(data.get('annotations', [])),
        'num_categories': len(data.get('categories', [])),
        'image_ids': set(img['id'] for img in data.get('images', [])),
        'image_filenames': set(img['file_name'] for img in data.get('images', [])),
        'categories': {cat['id']: cat['name'] for cat in data.get('categories', [])}
    }
    
    # Count annotations per category
    category_counts = defaultdict(int)
    corner_annotations = defaultdict(list)
    corner_categories = {'top_left_corner', 'top_right_corner', 'bottom_left_corner', 'bottom_right_corner'}
    
    for ann in data.get('annotations', []):
        cat_id = ann['category_id']
        cat_name = analysis['categories'].get(cat_id)
        category_counts[cat_id] += 1
        
        # Check if this is a corner annotation
        if cat_name in corner_categories:
            if 'keypoints' in ann:
                corner_annotations[cat_name].append({
                    'image_id': ann['image_id'],
                    'keypoints': ann['keypoints']
                })
            elif 'bbox' in ann:
                corner_annotations[cat_name].append({
                    'image_id': ann['image_id'],
                    'bbox': ann['bbox']
                })
    
    analysis['annotations_per_category'] = dict(category_counts)
    analysis['corner_annotations'] = corner_annotations
    
    return analysis

def find_duplicates(dataset1, dataset2):
    """Find duplicate images between two datasets based on filenames."""
    if not dataset1 or not dataset2:
        return set()
    
    files1 = set(img['file_name'] for img in dataset1.get('images', []))
    files2 = set(img['file_name'] for img in dataset2.get('images', []))
    
    return files1.intersection(files2)

def print_corner_analysis(analysis, dataset_name):
    """Print detailed analysis of corner annotations."""
    print(f"\n=== {dataset_name} Corner Analysis ===")
    corner_data = analysis.get('corner_annotations', {})
    if not any(corner_data.values()):
        print("No corner annotations (keypoints or bounding boxes) found")
        return
        
    for corner_type, annotations in corner_data.items():
        print(f"\n{corner_type}:")
        print(f"  Total annotations: {len(annotations)}")
        if annotations:
            print("  Annotation type:", "keypoints" if 'keypoints' in annotations[0] else "bbox")
            for idx, ann in enumerate(annotations[:3], 1):
                if 'keypoints' in ann:
                    print(f"  Example {idx}: Image ID {ann['image_id']}, Keypoints: {ann['keypoints']}")
                else:
                    print(f"  Example {idx}: Image ID {ann['image_id']}, BBox: {ann['bbox']}")
            if len(annotations) > 3:
                print(f"  ... and {len(annotations) - 3} more annotations")

def main():
    # Define paths
    abenathi_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\result.json"
    dj_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset\result.json"
    
    # Load datasets
    print("\nLoading datasets...")
    abenathi_data = load_coco_json(abenathi_path)
    dj_data = load_coco_json(dj_path)
    
    # Analyze datasets
    print("\nAnalyzing Abenathi's dataset...")
    abenathi_analysis = analyze_dataset(abenathi_data, "Abenathi's dataset")
    
    print("\nAnalyzing DJ's dataset...")
    dj_analysis = analyze_dataset(dj_data, "DJ's dataset")
    
    # Print analysis results
    for name, analysis in [("Abenathi's dataset", abenathi_analysis), ("DJ's dataset", dj_analysis)]:
        if analysis:
            print(f"\n=== {name} Analysis ===")
            print(f"Number of images: {analysis['num_images']}")
            print(f"Number of annotations: {analysis['num_annotations']}")
            print(f"Number of categories: {analysis['num_categories']}")
            print("\nCategories:")
            for cat_id, cat_name in analysis['categories'].items():
                count = analysis['annotations_per_category'].get(cat_id, 0)
                print(f"  - {cat_name} (ID: {cat_id}): {count} annotations")
    
    # Print corner analysis
    if abenathi_analysis:
        print_corner_analysis(abenathi_analysis, "Abenathi's dataset")
    if dj_analysis:
        print_corner_analysis(dj_analysis, "DJ's dataset")
    
    # Check for duplicates
    duplicates = find_duplicates(abenathi_data, dj_data)
    print("\n=== Duplicate Analysis ===")
    if duplicates:
        print(f"Found {len(duplicates)} duplicate filenames:")
        for filename in sorted(duplicates):
            print(f"  - {filename}")
    else:
        print("No duplicate filenames found between the datasets")

if __name__ == "__main__":
    main()
