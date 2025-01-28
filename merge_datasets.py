import json
import os
import shutil
from zipfile import ZipFile

def merge_datasets():
    # Create output directory
    os.makedirs("merged_dataset/images", exist_ok=True)
    
    # Process each dataset
    datasets = ["dataset1.zip", "dataset2.zip"]
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "New"},
            {"id": 2, "name": "Old"}
        ]
    }
    
    max_image_id = 0
    max_annotation_id = 0
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        
        # Extract zip
        with ZipFile(dataset) as zf:
            temp_dir = f"temp_{dataset.replace('.zip', '')}"
            os.makedirs(temp_dir, exist_ok=True)
            zf.extractall(temp_dir)
        
        # Load JSON
        json_path = os.path.join(temp_dir, "result.json")
        if not os.path.exists(json_path):
            print(f"No result.json found in {dataset}")
            continue
            
        with open(json_path) as f:
            data = json.load(f)
        
        # Process images and annotations
        for img in data["images"]:
            old_id = img["id"]
            new_id = old_id + max_image_id
            img["id"] = new_id
            
            # Copy image
            src = os.path.join(temp_dir, "images", os.path.basename(img["file_name"]))
            dst = os.path.join("merged_dataset/images", os.path.basename(img["file_name"]))
            if os.path.exists(src):
                shutil.copy2(src, dst)
                merged_data["images"].append(img)
            
            # Update annotations
            for ann in data["annotations"]:
                if ann["image_id"] == old_id:
                    ann["image_id"] = new_id
                    ann["id"] += max_annotation_id
                    merged_data["annotations"].append(ann)
        
        # Update max IDs
        if data["images"]:
            max_image_id = max(img["id"] for img in merged_data["images"])
        if merged_data["annotations"]:
            max_annotation_id = max(ann["id"] for ann in merged_data["annotations"])
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    # Save merged dataset
    with open("merged_dataset/merged_coco.json", "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\nMerged dataset created:")
    print(f"Images: {len(merged_data['images'])}")
    print(f"Annotations: {len(merged_data['annotations'])}")

if __name__ == "__main__":
    merge_datasets() 