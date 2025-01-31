import json
import os
import shutil
import random
from pathlib import Path

def split_dataset(merged_json_path, output_dir, train_ratio=0.8):
    """Split a COCO dataset into training and validation sets."""
    # Load merged dataset
    with open(merged_json_path) as f:
        data = json.load(f)
    
    # Get all available images from source directories
    source_images = {}
    seen_filenames = set()  # Track unique filenames
    for dataset_path in ["abenathi_object_detection_dataset", "dj_object_detection_dataset"]:
        img_dir = os.path.join(dataset_path, "images")
        if os.path.exists(img_dir):
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in Path(img_dir).glob(ext):
                    filename = img_path.name.lower()
                    if filename not in seen_filenames:
                        source_images[filename] = img_path
                        seen_filenames.add(filename)
    
    print(f"\nFound {len(source_images)} unique source images")
    
    # Filter out images that don't exist in source directories and deduplicate
    available_images = []
    available_image_ids = set()
    seen_filenames = set()
    for img in data["images"]:
        filename = img["file_name"].lower()
        if filename in source_images and filename not in seen_filenames:
            available_images.append(img)
            available_image_ids.add(img["id"])
            seen_filenames.add(filename)
    
    print(f"Found {len(available_images)} unique images in annotations with matching source files")
    
    # Get annotations for available images
    available_annotations = [ann for ann in data["annotations"] if ann["image_id"] in available_image_ids]
    print(f"Found {len(available_annotations)} annotations for available images")
    
    # Shuffle and split available images
    random.shuffle(available_images)
    num_train = int(len(available_images) * train_ratio)
    train_images = available_images[:num_train]
    val_images = available_images[num_train:]
    
    # Get image IDs for each set
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    
    # Split annotations
    train_annotations = [ann for ann in available_annotations if ann["image_id"] in train_ids]
    val_annotations = [ann for ann in available_annotations if ann["image_id"] in val_ids]
    
    # Create output datasets
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": data["categories"]
    }
    
    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": data["categories"]
    }
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    train_images_dir = os.path.join(train_dir, "images")
    val_images_dir = os.path.join(val_dir, "images")
    
    # Remove existing directories if they exist
    if os.path.exists(train_images_dir):
        shutil.rmtree(train_images_dir)
    if os.path.exists(val_images_dir):
        shutil.rmtree(val_images_dir)
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    
    # Save JSON files
    with open(os.path.join(train_dir, "annotations.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(val_dir, "annotations.json"), "w") as f:
        json.dump(val_data, f, indent=2)
    
    # Copy images to respective directories
    for img in train_images:
        source_img = source_images[img["file_name"].lower()]
        shutil.copy2(source_img, os.path.join(train_images_dir, img["file_name"]))
    
    for img in val_images:
        source_img = source_images[img["file_name"].lower()]
        shutil.copy2(source_img, os.path.join(val_images_dir, img["file_name"]))
    
    print("\nDataset split complete:")
    print(f"Training set: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Validation set: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations")
    print(f"\nCategories ({len(data['categories'])}):")
    for cat in data["categories"]:
        print(f"- {cat['name']} (id: {cat['id']})")

if __name__ == "__main__":
    merged_json_path = "merged_dataset.json"
    output_dir = "merged_dataset"
    split_dataset(merged_json_path, output_dir)
