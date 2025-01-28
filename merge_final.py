import json
import os
import shutil
from zipfile import ZipFile
from pathlib import Path

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_dataset(zip_path, extract_dir):
    """Extract dataset and return path to JSON file"""
    print(f"Extracting {zip_path} to {extract_dir}")
    with ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    
    json_path = os.path.join(extract_dir, "result.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"result.json not found in {extract_dir}")
    return json_path

def copy_image(src_dir, dst_dir, filename):
    """Copy image file and return success status"""
    src = os.path.join(src_dir, filename)
    dst = os.path.join(dst_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {filename}")
        return True
    print(f"Warning: Image not found - {src}")
    return False

def main():
    # Setup directories
    output_dir = "merged_output"
    ensure_dir(output_dir)
    images_dir = os.path.join(output_dir, "images")
    ensure_dir(images_dir)
    
    # Extract datasets
    temp_dir1 = "temp_dataset1"
    temp_dir2 = "temp_dataset2"
    ensure_dir(temp_dir1)
    ensure_dir(temp_dir2)
    
    try:
        # Load and process datasets
        json_path1 = extract_dataset("dataset1.zip", temp_dir1)
        json_path2 = extract_dataset("dataset2.zip", temp_dir2)
        
        with open(json_path1) as f:
            data1 = json.load(f)
        with open(json_path2) as f:
            data2 = json.load(f)
        
        # Initialize merged data
        merged_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "New"},
                {"id": 2, "name": "Old"}
            ]
        }
        
        # Process first dataset
        print("\nProcessing first dataset...")
        max_image_id = 0
        max_annotation_id = 0
        
        for img in data1["images"]:
            old_id = img["id"]
            img["id"] = old_id + max_image_id
            
            if copy_image(os.path.join(temp_dir1, "images"), images_dir, os.path.basename(img["file_name"])):
                merged_data["images"].append(img)
                
                # Add corresponding annotations
                for ann in data1["annotations"]:
                    if ann["image_id"] == old_id:
                        ann["image_id"] = img["id"]
                        ann["id"] += max_annotation_id
                        merged_data["annotations"].append(ann)
        
        # Update max IDs
        if merged_data["images"]:
            max_image_id = max(img["id"] for img in merged_data["images"])
        if merged_data["annotations"]:
            max_annotation_id = max(ann["id"] for ann in merged_data["annotations"])
        
        # Process second dataset
        print("\nProcessing second dataset...")
        for img in data2["images"]:
            old_id = img["id"]
            img["id"] = old_id + max_image_id
            
            if copy_image(os.path.join(temp_dir2, "images"), images_dir, os.path.basename(img["file_name"])):
                merged_data["images"].append(img)
                
                # Add corresponding annotations
                for ann in data2["annotations"]:
                    if ann["image_id"] == old_id:
                        ann["image_id"] = img["id"]
                        ann["id"] += max_annotation_id
                        merged_data["annotations"].append(ann)
        
        # Save merged dataset
        output_json = os.path.join(output_dir, "merged_coco.json")
        with open(output_json, "w") as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"\nMerged dataset statistics:")
        print(f"Total images: {len(merged_data['images'])}")
        print(f"Total annotations: {len(merged_data['annotations'])}")
        print(f"\nOutput saved to: {output_dir}")
        
    finally:
        # Cleanup
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)

if __name__ == "__main__":
    main() 