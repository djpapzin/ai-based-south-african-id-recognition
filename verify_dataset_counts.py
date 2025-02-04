import os
import json
from pathlib import Path

def verify_image_counts():
    """Verify total image counts match between original and split datasets"""
    # Get project root directory
    project_dir = Path(__file__).parent
    
    # Define paths
    original_dir = project_dir / "original_dataset"
    train_dir = project_dir / "merged_dataset" / "train" / "images"
    val_dir = project_dir / "merged_dataset" / "val" / "images"
    
    # Count original images if directory exists
    if original_dir.exists():
        original_images = set(f for f in os.listdir(original_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        print(f"\nOriginal dataset: {len(original_images)} images")
    else:
        print(f"\nWarning: Original dataset directory not found at {original_dir}")
        original_images = set()
    
    # Count training images
    if train_dir.exists():
        train_images = set(f for f in os.listdir(train_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        print(f"Training set: {len(train_images)} images")
    else:
        print(f"Warning: Training directory not found at {train_dir}")
        train_images = set()
    
    # Count validation images
    if val_dir.exists():
        val_images = set(f for f in os.listdir(val_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        print(f"Validation set: {len(val_images)} images")
    else:
        print(f"Warning: Validation directory not found at {val_dir}")
        val_images = set()
    
    # Check for matches if original dataset exists
    if original_images:
        missing_in_splits = original_images - train_images - val_images
        if missing_in_splits:
            print(f"\nWarning: {len(missing_in_splits)} images missing from splits:")
            for img in sorted(missing_in_splits)[:3]:  # Show first 3 missing
                print(f" - {img}")
            if len(missing_in_splits) > 3:
                print(f" ...and {len(missing_in_splits)-3} more")
    
    # Check annotation counts
    train_ann_path = project_dir / "merged_dataset" / "train" / "annotations.json"
    val_ann_path = project_dir / "merged_dataset" / "val" / "annotations.json"
    
    print(f"\nAnnotations report:")
    if train_ann_path.exists():
        with open(train_ann_path) as f:
            train_ann = json.load(f)
            print(f" - Training annotations: {len(train_ann['images'])} registered images")
            print(f" - Training annotations: {len(train_ann['annotations'])} total annotations")
    else:
        print(" - Training annotations file not found")
    
    if val_ann_path.exists():
        with open(val_ann_path) as f:
            val_ann = json.load(f)
            print(f" - Validation annotations: {len(val_ann['images'])} registered images")
            print(f" - Validation annotations: {len(val_ann['annotations'])} total annotations")
    else:
        print(" - Validation annotations file not found")
    
    # Print totals
    total_in_splits = len(train_images) + len(val_images)
    if original_images:
        print(f"\nTotal images:")
        print(f" - Original: {len(original_images)}")
        print(f" - In splits: {total_in_splits}")
        print(f" - Difference: {len(original_images) - total_in_splits}")
    else:
        print(f"\nTotal images in splits: {total_in_splits}")

if __name__ == "__main__":
    print("Verifying dataset counts...")
    verify_image_counts()
    print("\nVerification complete!")
