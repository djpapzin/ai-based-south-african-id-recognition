import os
import json
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset(merged_dataset_path, output_base_path, val_split=0.2, random_state=42):
    """
    Prepare the merged dataset by splitting it into train and validation sets.
    
    Args:
        merged_dataset_path (str): Path to the merged dataset directory
        output_base_path (str): Base path for output directories
        val_split (float): Fraction of data to use for validation
        random_state (int): Random seed for reproducibility
    """
    # Create output directories
    train_dir = os.path.join(output_base_path, "train")
    val_dir = os.path.join(output_base_path, "val")
    
    for dir_path in [train_dir, val_dir]:
        os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
    
    # Load merged annotations
    with open(os.path.join(merged_dataset_path, "result.json"), 'r') as f:
        merged_data = json.load(f)
    
    # Split images
    image_ids = [img['id'] for img in merged_data['images']]
    train_ids, val_ids = train_test_split(image_ids, test_size=val_split, random_state=random_state)
    
    # Prepare train and val datasets
    train_data = {
        "images": [],
        "annotations": [],
        "categories": merged_data['categories']
    }
    
    val_data = {
        "images": [],
        "annotations": [],
        "categories": merged_data['categories']
    }
    
    # Process images and copy files
    for img in merged_data['images']:
        src_path = os.path.join(merged_dataset_path, "images", img['file_name'])
        
        if img['id'] in train_ids:
            dst_path = os.path.join(train_dir, "images", img['file_name'])
            train_data['images'].append(img)
        else:
            dst_path = os.path.join(val_dir, "images", img['file_name'])
            val_data['images'].append(img)
            
        shutil.copy2(src_path, dst_path)
    
    # Process annotations
    for ann in merged_data['annotations']:
        if ann['image_id'] in train_ids:
            train_data['annotations'].append(ann)
        else:
            val_data['annotations'].append(ann)
    
    # Save annotations
    with open(os.path.join(train_dir, "annotations.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(val_dir, "annotations.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Dataset split complete:")
    print(f"Training: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Validation: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    
    return train_dir, val_dir

if __name__ == "__main__":
    # Example usage
    merged_dataset_path = "merged_object_detection_dataset"
    output_base_path = "split_dataset"
    train_dir, val_dir = prepare_dataset(merged_dataset_path, output_base_path) 