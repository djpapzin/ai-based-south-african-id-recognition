import json
import os
from pprint import pprint
from collections import defaultdict

def analyze_dataset(dataset_path, name):
    """Analyze a single dataset."""
    print(f"\n=== Analyzing {name} dataset ===")
    print(f"Path: {dataset_path}")
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {dataset_path}")
        return None
    
    # Basic statistics
    print(f"\nStatistics:")
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {len(data['categories'])}")
    
    # Analyze categories
    print("\nCategories and their annotation counts:")
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    for cat_id, name in sorted(categories.items()):
        count = len([a for a in data['annotations'] if a['category_id'] == cat_id])
        print(f"- {name} (id: {cat_id}): {count} annotations")
    
    # Distribution of annotations per image
    annotations_per_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_per_image[ann['image_id']].append(ann['category_id'])
    
    count_distribution = defaultdict(int)
    for image_id, category_ids in annotations_per_image.items():
        count_distribution[len(category_ids)] += 1
    
    print("\nDistribution of annotations per image:")
    for count, freq in sorted(count_distribution.items()):
        print(f"- {count} annotations: {freq} images")
    
    return data

def compare_datasets():
    """Compare both datasets."""
    dj_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset\result.json"
    abenathi_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\result.json"
    
    dj_data = analyze_dataset(dj_path, "DJ")
    abenathi_data = analyze_dataset(abenathi_path, "Abenathi")
    
    if dj_data and abenathi_data:
        print("\n=== Comparing Datasets ===")
        
        # Compare categories
        dj_categories = {cat['name']: cat['id'] for cat in dj_data['categories']}
        abenathi_categories = {cat['name']: cat['id'] for cat in abenathi_data['categories']}
        
        print("\nCategory comparison:")
        all_categories = sorted(set(dj_categories.keys()) | set(abenathi_categories.keys()))
        for cat in all_categories:
            in_dj = cat in dj_categories
            in_abenathi = cat in abenathi_categories
            if in_dj and in_abenathi:
                print(f"- {cat}: Present in both datasets")
            elif in_dj:
                print(f"- {cat}: Only in DJ dataset")
            else:
                print(f"- {cat}: Only in Abenathi dataset")
        
        # Compare annotation structure
        print("\nAnnotation structure comparison:")
        dj_ann = dj_data['annotations'][0] if dj_data['annotations'] else None
        abenathi_ann = abenathi_data['annotations'][0] if abenathi_data['annotations'] else None
        
        if dj_ann and abenathi_ann:
            dj_fields = set(dj_ann.keys())
            abenathi_fields = set(abenathi_ann.keys())
            
            print("\nFields in both datasets:")
            for field in sorted(dj_fields & abenathi_fields):
                print(f"- {field}")
            
            print("\nFields only in DJ dataset:")
            for field in sorted(dj_fields - abenathi_fields):
                print(f"- {field}")
            
            print("\nFields only in Abenathi dataset:")
            for field in sorted(abenathi_fields - dj_fields):
                print(f"- {field}")

if __name__ == "__main__":
    compare_datasets()
