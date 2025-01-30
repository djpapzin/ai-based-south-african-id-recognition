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

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_images_dir, exist_ok=True)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def copy_images(src_dir, dest_dir, prefix=""):
    """Copy images and return a mapping of old to new filenames"""
    src_images_dir = os.path.join(src_dir, "images")
    filename_mapping = {}
    
    for img in os.listdir(src_images_dir):
        src_path = os.path.join(src_images_dir, img)
        if os.path.isfile(src_path):
            # Add prefix to filename if provided
            new_name = f"{prefix}{img}" if prefix else img
            dest_path = os.path.join(dest_dir, new_name)
            
            # Handle duplicates by adding a counter
            base, ext = os.path.splitext(new_name)
            counter = 1
            while os.path.exists(dest_path):
                new_name = f"{base}_{counter}{ext}"
                dest_path = os.path.join(dest_dir, new_name)
                counter += 1
            
            shutil.copy2(src_path, dest_path)
            filename_mapping[img] = new_name
    
    return filename_mapping

def update_annotations(annotations, filename_mapping):
    """Update image filenames in annotations"""
    updated_annotations = []
    
    for ann in annotations:
        if isinstance(ann, dict) and 'image' in ann:
            old_filename = os.path.basename(ann['image'])
            if old_filename in filename_mapping:
                ann_copy = ann.copy()
                ann_copy['image'] = filename_mapping[old_filename]
                updated_annotations.append(ann_copy)
            else:
                updated_annotations.append(ann)
        else:
            updated_annotations.append(ann)
    
    return updated_annotations

def merge_datasets():
    # Load annotations
    abenathi_annotations = load_json(os.path.join(abenathi_dir, "result.json"))
    dj_annotations = load_json(os.path.join(dj_dir, "result.json"))
    
    # Copy images and get filename mappings
    print("Copying Abenathi's images...")
    abenathi_mapping = copy_images(abenathi_dir, output_images_dir, prefix="abenathi_")
    print("Copying DJ's images...")
    dj_mapping = copy_images(dj_dir, output_images_dir, prefix="dj_")
    
    # Update annotations with new filenames
    print("Updating annotations...")
    updated_abenathi = update_annotations(abenathi_annotations, abenathi_mapping)
    updated_dj = update_annotations(dj_annotations, dj_mapping)
    
    # Merge annotations
    merged_annotations = updated_abenathi + updated_dj
    
    # Save merged annotations
    output_json = os.path.join(output_dir, "result.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerge completed!")
    print(f"Total annotations: {len(merged_annotations)}")
    print(f"Images from Abenathi's dataset: {len(abenathi_mapping)}")
    print(f"Images from DJ's dataset: {len(dj_mapping)}")
    print(f"\nMerged dataset saved to: {output_dir}")

if __name__ == "__main__":
    merge_datasets() 