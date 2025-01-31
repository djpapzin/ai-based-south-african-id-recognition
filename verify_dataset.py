import json
import os
from collections import defaultdict
from typing import Dict, List, Set

# Define expected labels for each ID type
NEW_ID_LABELS = {
    'id_document', 'surname', 'names', 'sex', 'nationality', 'id_number',
    'date_of_birth', 'country_of_birth', 'citizenship_status', 'face',
    'signature', 'top_left_corner', 'top_right_corner', 'bottom_left_corner',
    'bottom_right_corner'
}

OLD_ID_LABELS = {
    'id_document', 'surname', 'names', 'id_number', 'date_of_birth',
    'country_of_birth', 'citizenship_status', 'face', 'top_left_corner',
    'top_right_corner', 'bottom_left_corner', 'bottom_right_corner'
}

def load_json_file(file_path: str) -> dict:
    """Load and return JSON file content."""
    with open(file_path, 'r') as f:
        return json.load(f)

def identify_id_type(labels: Set[str]) -> str:
    """
    Identify if the image is more likely to be a new or old ID based on its labels.
    """
    # If it has sex, nationality, or signature, it's likely a new ID
    new_id_specific = {'sex', 'nationality', 'signature'}
    if any(label in labels for label in new_id_specific):
        return 'new'
    return 'old'

def check_duplicate_labels(annotations: dict, dataset_name: str) -> Dict[str, List[str]]:
    """
    Check for duplicate labels in each image.
    Returns a dictionary of image_id: list of duplicate categories
    """
    issues = {}
    
    # Group annotations by image_id
    image_annotations = defaultdict(lambda: defaultdict(int))
    
    # Create mapping of image_id to filename for better reporting
    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
    
    # Count occurrences of each category per image
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_annotations[image_id][category_id] += 1
    
    # Check for duplicates
    for image_id, category_counts in image_annotations.items():
        duplicates = []
        for category_id, count in category_counts.items():
            if count > 1:
                # Get category name
                category_name = next(
                    (cat['name'] for cat in annotations['categories'] 
                     if cat['id'] == category_id), 
                    f'Unknown Category {category_id}'
                )
                duplicates.append(f"{category_name} ({count} instances)")
        
        if duplicates:
            filename = image_id_to_filename[image_id]
            issues[filename] = duplicates
    
    return issues

def check_labels(annotations: dict, dataset_name: str):
    """Check labels against expected counts for new and old IDs."""
    
    # Get category map
    category_map = {cat['id']: cat['name'] for cat in annotations['categories']}
    
    # Group annotations by image
    image_annotations = defaultdict(set)
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        category_name = category_map[ann['category_id']]
        image_annotations[image_id].add(category_name)
    
    # Check each image
    issues = []
    for img in annotations['images']:
        img_id = img['id']
        img_labels = image_annotations[img_id]
        
        # Identify ID type
        id_type = identify_id_type(img_labels)
        expected_labels = NEW_ID_LABELS if id_type == 'new' else OLD_ID_LABELS
        
        # Check for missing labels
        missing = expected_labels - img_labels
        
        if missing:
            issues.append({
                'filename': img['file_name'],
                'id_type': id_type,
                'current_count': len(img_labels),
                'expected_count': len(expected_labels),
                'missing': sorted(list(missing))
            })
    
    # Sort issues by number of missing labels
    issues.sort(key=lambda x: len(x['missing']), reverse=True)
    
    if issues:
        print(f"\n{dataset_name}: Found {len(issues)} images with missing labels:")
        
        print("\nNew ID Documents with missing labels:")
        for issue in [i for i in issues if i['id_type'] == 'new']:
            print(f"\nImage: {issue['filename']}")
            print(f"Current labels: {issue['current_count']}/15")
            print("Missing labels:")
            for label in issue['missing']:
                print(f"  - {label}")
        
        print("\nOld ID Documents with missing labels:")
        for issue in [i for i in issues if i['id_type'] == 'old']:
            print(f"\nImage: {issue['filename']}")
            print(f"Current labels: {issue['current_count']}/12")
            print("Missing labels:")
            for label in issue['missing']:
                print(f"  - {label}")
    else:
        print(f"\n{dataset_name}: All images have the correct number of labels! ")

def check_label_counts(annotations: dict, dataset_name: str):
    """Check if each image has exactly 15 labels and report discrepancies."""
    
    # Get all category names
    category_map = {cat['id']: cat['name'] for cat in annotations['categories']}
    expected_categories = set(category_map.values())
    
    # Group annotations by image
    image_annotations = defaultdict(list)
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        category_name = category_map[ann['category_id']]
        image_annotations[image_id].append(category_name)
    
    # Check each image
    issues = []
    for img in annotations['images']:
        img_id = img['id']
        img_labels = set(image_annotations[img_id])
        
        if len(img_labels) != 15:
            missing = expected_categories - img_labels
            extra = img_labels - expected_categories
            
            issues.append({
                'filename': img['file_name'],
                'label_count': len(img_labels),
                'missing': sorted(list(missing)),
                'extra': sorted(list(extra))
            })
    
    # Sort issues by number of labels (most problematic first)
    issues.sort(key=lambda x: abs(15 - x['label_count']))
    
    if issues:
        print(f"\n{dataset_name}: Found {len(issues)} images without exactly 15 labels:")
        for issue in issues:
            print(f"\nImage: {issue['filename']}")
            print(f"Total labels: {issue['label_count']}")
            if issue['missing']:
                print("Missing labels:")
                for label in issue['missing']:
                    print(f"  - {label}")
            if issue['extra']:
                print("Extra labels:")
                for label in issue['extra']:
                    print(f"  - {label}")
    else:
        print(f"\n{dataset_name}: All images have exactly 15 labels! ")

def analyze_dataset(annotations: dict, dataset_name: str):
    """Analyze dataset and print statistics"""
    print(f"\nAnalyzing {dataset_name}...")
    
    # Get category distribution
    category_counts = defaultdict(int)
    for ann in annotations['annotations']:
        category_counts[ann['category_id']] += 1
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total images: {len(annotations['images'])}")
    print(f"Total annotations: {len(annotations['annotations'])}")
    print("\nCategories:")
    for cat in annotations['categories']:
        count = category_counts[cat['id']]
        print(f"  - {cat['name']}: {count} instances")
    
    # Check for images without annotations
    annotated_images = set(ann['image_id'] for ann in annotations['annotations'])
    all_images = set(img['id'] for img in annotations['images'])
    unannotated_images = all_images - annotated_images
    
    if unannotated_images:
        print(f"\nWarning: Found {len(unannotated_images)} images without annotations:")
        for img_id in unannotated_images:
            img_name = next(img['file_name'] for img in annotations['images'] if img['id'] == img_id)
            print(f"  - {img_name}")

def main():
    # Dataset paths
    datasets = {
        "Abenathi Dataset": "C:/Users/lfana/Documents/Kwantu/Machine Learning/abenathi_object_detection_dataset/result.json",
        "DJ Dataset": "C:/Users/lfana/Documents/Kwantu/Machine Learning/dj_object_detection_dataset/result.json"
    }
    
    for dataset_name, dataset_path in datasets.items():
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} not found!")
            continue
            
        print(f"\nChecking {dataset_name}...")
        try:
            annotations = load_json_file(dataset_path)
            
            # Check for duplicate labels
            issues = check_duplicate_labels(annotations, dataset_name)
            if issues:
                print(f"\nFound {len(issues)} images with duplicate labels:")
                for filename, duplicates in issues.items():
                    print(f"\nImage: {filename}")
                    print("Duplicate categories:")
                    for dup in duplicates:
                        print(f"  - {dup}")
            else:
                print("No duplicate labels found! ")
            
            # Check for missing or extra labels
            check_label_counts(annotations, dataset_name)
            
            # Check labels against expected counts for new and old IDs
            check_labels(annotations, dataset_name)
            
            # Analyze dataset statistics
            analyze_dataset(annotations, dataset_name)
                
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

if __name__ == "__main__":
    main()
