import json
import os
import shutil
from pathlib import Path

# Input paths
abenathi_dir = "abenathi_object_detection_dataset"
dj_dir = "dj_object_detection_dataset"

# Output paths
output_dir = "merged_object_detection_dataset"
output_images_dir = os.path.join(output_dir, "images")

def load_json(file_path):
    """Load and validate JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not all(key in data for key in ['images', 'annotations', 'categories']):
            raise ValueError(f"Missing required keys in {file_path}")
        return data

def validate_coco_format(data):
    """Validate that the data follows COCO format"""
    required_keys = ['images', 'categories', 'annotations']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in dataset")
    
    # Validate images
    for img in data['images']:
        if not all(k in img for k in ['id', 'file_name']):
            raise ValueError("Image missing required fields (id, file_name)")
    
    # Validate categories
    for cat in data['categories']:
        if not all(k in cat for k in ['id', 'name']):
            raise ValueError("Category missing required fields (id, name)")
    
    # Validate annotations
    for ann in data['annotations']:
        if not all(k in ann for k in ['id', 'image_id', 'category_id', 'bbox']):
            raise ValueError("Annotation missing required fields (id, image_id, category_id, bbox)")

def copy_images(src_dir, dest_dir, prefix=""):
    """Copy images and return a mapping of old to new filenames"""
    src_images_dir = os.path.join(src_dir, "images")
    filename_mapping = {}
    
    # Ensure output directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    for img in os.listdir(src_images_dir):
        src_path = os.path.join(src_images_dir, img)
        if os.path.isfile(src_path):
            # Add prefix to filename if provided
            new_name = f"{prefix}{img}" if prefix else img
            dest_path = os.path.join(dest_dir, new_name)
            
            # Copy file if it doesn't exist
            if not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {img} -> {new_name}")
            filename_mapping[img] = new_name
    
    return filename_mapping

def update_image_data(images, filename_mapping, id_offset=0):
    """Update image data with new filenames and adjusted IDs"""
    updated_images = []
    id_mapping = {}  # Map old IDs to new IDs
    
    for img in images:
        old_filename = os.path.basename(img['file_name'].replace('images\\', '').replace('images/', ''))
        if old_filename in filename_mapping:
            img_copy = img.copy()
            img_copy['file_name'] = filename_mapping[old_filename]
            old_id = img_copy['id']
            new_id = old_id + id_offset
            img_copy['id'] = new_id
            id_mapping[old_id] = new_id
            updated_images.append(img_copy)
            print(f"Updated image ID: {old_id} -> {new_id}")
    
    return updated_images, id_mapping

def validate_annotation(ann):
    """Validate a single annotation"""
    required_fields = ['id', 'image_id', 'category_id', 'bbox']
    if not all(field in ann for field in required_fields):
        raise ValueError(f"Annotation missing required fields: {required_fields}")
    if not isinstance(ann['bbox'], list) or len(ann['bbox']) != 4:
        raise ValueError(f"Invalid bbox format in annotation {ann['id']}")
    return True

def update_annotations(annotations, id_mapping, annotation_id_offset=0):
    """Update annotation IDs and their image references"""
    updated_annotations = []
    current_id = annotation_id_offset
    
    for ann in annotations:
        try:
            validate_annotation(ann)
            if ann['image_id'] in id_mapping:
                ann_copy = ann.copy()
                # Update image reference
                ann_copy['image_id'] = id_mapping[ann['image_id']]
                # Assign new unique ID
                ann_copy['id'] = current_id
                current_id += 1
                # Ensure bbox is properly formatted
                ann_copy['bbox'] = [float(x) for x in ann_copy['bbox']]
                # Add area if missing
                if 'area' not in ann_copy:
                    ann_copy['area'] = ann_copy['bbox'][2] * ann_copy['bbox'][3]
                # Add iscrowd if missing
                if 'iscrowd' not in ann_copy:
                    ann_copy['iscrowd'] = 0
                
                updated_annotations.append(ann_copy)
                print(f"Updated annotation: image_id {ann['image_id']} -> {ann_copy['image_id']}, id {ann['id']} -> {ann_copy['id']}")
        except ValueError as e:
            print(f"Warning: Skipping invalid annotation: {e}")
            continue
            
    return updated_annotations

def merge_datasets():
    # Clean output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(output_images_dir)
    
    print("\nLoading and validating datasets...")
    
    # Load and validate annotations
    abenathi_data = load_json(os.path.join(abenathi_dir, "result.json"))
    dj_data = load_json(os.path.join(dj_dir, "result.json"))
    
    validate_coco_format(abenathi_data)
    validate_coco_format(dj_data)
    
    print("\nCopying and updating images...")
    
    # Copy images and get filename mappings
    abenathi_mapping = copy_images(abenathi_dir, output_images_dir, prefix="abenathi_")
    dj_mapping = copy_images(dj_dir, output_images_dir, prefix="dj_")
    
    print("\nUpdating image references...")
    
    # Update and merge images
    abenathi_images, abenathi_id_mapping = update_image_data(abenathi_data['images'], abenathi_mapping)
    dj_images, dj_id_mapping = update_image_data(dj_data['images'], dj_mapping, id_offset=len(abenathi_images))
    
    merged_images = abenathi_images + dj_images
    
    print("\nMerging annotations...")
    
    # Update and merge annotations
    abenathi_annotations = update_annotations(
        abenathi_data['annotations'],
        abenathi_id_mapping
    )
    dj_annotations = update_annotations(
        dj_data['annotations'],
        dj_id_mapping,
        annotation_id_offset=len(abenathi_annotations)
    )
    
    merged_annotations = abenathi_annotations + dj_annotations
    
    print("\nMerging categories...")
    
    # Merge categories (removing duplicates while preserving IDs)
    category_map = {}
    merged_categories = []
    for cat in abenathi_data['categories'] + dj_data['categories']:
        if cat['name'] not in [c['name'] for c in merged_categories]:
            merged_categories.append(cat)
    
    # Create merged dataset
    merged_data = {
        "images": merged_images,
        "categories": merged_categories,
        "annotations": merged_annotations
    }
    
    # Validate merged dataset
    print("\nValidating merged dataset...")
    validate_coco_format(merged_data)
    
    # Save merged annotations
    output_json = os.path.join(output_dir, "result.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerge completed successfully!")
    print(f"Total images: {len(merged_images)}")
    print(f"Images from Abenathi's dataset: {len(abenathi_mapping)}")
    print(f"Images from DJ's dataset: {len(dj_mapping)}")
    print(f"Total annotations: {len(merged_annotations)}")
    print(f"Total categories: {len(merged_categories)}")
    print(f"\nMerged dataset saved to: {output_dir}")

if __name__ == "__main__":
    merge_datasets() 