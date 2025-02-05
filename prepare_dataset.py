import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image

def convert_and_save_image(src_path, dst_path):
    """Convert image to RGB JPG format and save it."""
    try:
        with Image.open(src_path) as img:
            # Convert RGBA to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])
                else:
                    background.paste(img, mask=img.split()[1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(dst_path, 'JPEG', quality=95, optimize=True)
            return True
    except Exception as e:
        print(f"Error converting {src_path}: {str(e)}")
        return False

def prepare_dataset(src_dir, train_dir, val_dir, split_ratio=0.8, seed=42):
    """Prepare dataset by splitting and converting images."""
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get list of all images
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle and split
    random.shuffle(image_files)
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Process training images
    print("\nProcessing training images...")
    train_success = 0
    for img_file in train_files:
        src_path = os.path.join(src_dir, img_file)
        dst_path = os.path.join(train_dir, Path(img_file).stem + '.jpg')
        
        if convert_and_save_image(src_path, dst_path):
            train_success += 1
            print(f"Converted: {img_file}")
        else:
            print(f"Failed to convert: {img_file}")
    
    # Process validation images
    print("\nProcessing validation images...")
    val_success = 0
    for img_file in val_files:
        src_path = os.path.join(src_dir, img_file)
        dst_path = os.path.join(val_dir, Path(img_file).stem + '.jpg')
        
        if convert_and_save_image(src_path, dst_path):
            val_success += 1
            print(f"Converted: {img_file}")
        else:
            print(f"Failed to convert: {img_file}")
    
    print(f"\nSuccessfully processed:")
    print(f"Training: {train_success}/{len(train_files)}")
    print(f"Validation: {val_success}/{len(val_files)}")
    
    return train_files, val_files

def update_dataset_categories(dataset_dir):
    """
    Update the dataset annotations to include all ID document fields and keypoints.
    Filter out back ID images and annotations.
    
    Args:
        dataset_dir (str): Path to the dataset directory (train or val)
    """
    annotations_file = os.path.join(dataset_dir, "annotations.json")
    
    # Read existing annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # Define all categories for ID document fields
    categories = [
        {"id": 0, "name": "id_document", "supercategory": "document"},
        {"id": 1, "name": "surname", "supercategory": "field"},
        {"id": 2, "name": "names", "supercategory": "field"},
        {"id": 3, "name": "sex", "supercategory": "field"},
        {"id": 4, "name": "nationality", "supercategory": "field"},
        {"id": 5, "name": "id_number", "supercategory": "field"},
        {"id": 6, "name": "date_of_birth", "supercategory": "field"},
        {"id": 7, "name": "country_of_birth", "supercategory": "field"},
        {"id": 8, "name": "citizenship_status", "supercategory": "field"},
        {"id": 9, "name": "face", "supercategory": "field"},
        {"id": 10, "name": "signature", "supercategory": "field"}
    ]
    
    # Define keypoint names
    keypoint_names = [
        "top_left_corner",
        "top_right_corner",
        "bottom_right_corner",
        "bottom_left_corner"
    ]
    
    # Update categories in the data
    data['categories'] = categories
    
    # Add keypoint information to the dataset
    data['keypoint_names'] = keypoint_names
    data['keypoint_flip_map'] = []  # No flipping for document corners
    data['keypoint_connection_rules'] = [
        (0, 1, (102, 204, 255)),  # top-left to top-right
        (1, 2, (102, 204, 255)),  # top-right to bottom-right
        (2, 3, (102, 204, 255)),  # bottom-right to bottom-left
        (3, 0, (102, 204, 255))   # bottom-left to top-left
    ]
    
    # Filter out back ID images and their annotations
    front_images = []
    front_annotations = []
    back_image_ids = set()
    
    # First identify back images
    for img in data['images']:
        filename = img['file_name'].lower()
        if "_b." in filename or "_back." in filename:
            back_image_ids.add(img['id'])
        else:
            front_images.append(img)
    
    # Filter annotations for front images only
    for ann in data['annotations']:
        if ann['image_id'] not in back_image_ids:
            # Keep the original category_id if it exists
            if 'category_id' not in ann:
                # If no category_id, assume it's the main id_document
                ann['category_id'] = 0
            
            # Ensure keypoints are properly formatted if they exist
            if 'keypoints' in ann:
                # Make sure we have exactly 4 keypoints (12 values - x,y,v for each point)
                if len(ann['keypoints']) != 12:
                    print(f"Warning: Annotation {ann['id']} has incorrect number of keypoint values")
                    continue
                
                # Ensure num_keypoints is set correctly
                ann['num_keypoints'] = 4
            
            front_annotations.append(ann)
    
    # Update the data with filtered content
    data['images'] = front_images
    data['annotations'] = front_annotations
    
    # Save updated annotations
    with open(annotations_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {dataset_dir}:")
    print(f"Front ID Images: {len(front_images)}")
    print(f"Front ID Annotations: {len(front_annotations)}")
    print(f"Categories: {[cat['name'] for cat in categories]}")
    print(f"Keypoints: {keypoint_names}")

def main():
    # Update both train and val datasets
    base_dir = "merged_dataset"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    print("Updating dataset categories...")
    update_dataset_categories(train_dir)
    update_dataset_categories(val_dir)
    print("\nDataset update complete!")

if __name__ == "__main__":
    # Prepare dataset
    src_dir = "dj_object_detection_dataset/images"
    train_dir = "merged_dataset/train/images"
    val_dir = "merged_dataset/val/images"
    
    # Clean output directories
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Prepare dataset
    print("Preparing dataset...")
    train_files, val_files = prepare_dataset(src_dir, train_dir, val_dir)
    
    # Run conversion script
    print("\nRunning conversion script...")
    os.system("python convert_to_coco.py") 