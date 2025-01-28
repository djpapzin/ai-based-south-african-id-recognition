import json
import os
import shutil
from zipfile import ZipFile
from pathlib import Path

def merge_coco_datasets(dataset1_path, dataset2_path, output_dir):
    """
    Merge two COCO format datasets
    
    Args:
        dataset1_path: Path to first dataset zip file
        dataset2_path: Path to second dataset zip file
        output_dir: Directory to store merged dataset
    """
    # Create output directory and images subdirectory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Dictionary to store merged data
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "New", "supercategory": "ID"},
            {"id": 2, "name": "Old", "supercategory": "ID"}
        ]
    }
    
    # Keep track of current max IDs
    max_image_id = 0
    max_annotation_id = 0
    
    # Process each dataset
    for dataset_path in [dataset1_path, dataset2_path]:
        print(f"\nProcessing {dataset_path}...")
        
        # Extract the zip file
        with ZipFile(dataset_path, 'r') as zip_ref:
            # Create a temporary directory for this dataset
            temp_dir = os.path.join(output_dir, f"temp_{Path(dataset_path).stem}")
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)
            
            # Find and load the COCO JSON file (either result.json or any .json file)
            json_file = None
            if os.path.exists(os.path.join(temp_dir, 'result.json')):
                json_file = 'result.json'
            else:
                json_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
                if json_files:
                    json_file = json_files[0]
            
            if not json_file:
                print(f"No JSON file found in {dataset_path}")
                continue
                
            print(f"Found JSON file: {json_file}")
            with open(os.path.join(temp_dir, json_file), 'r') as f:
                data = json.load(f)
            
            # Update image IDs and copy images
            for img in data['images']:
                old_id = img['id']
                new_id = old_id + max_image_id
                img['id'] = new_id
                
                # Copy image file to output directory
                img_filename = os.path.basename(img['file_name'])
                src_path = os.path.join(temp_dir, 'images', img_filename)
                dst_path = os.path.join(images_dir, img_filename)
                
                if os.path.exists(src_path):
                    print(f"Copying image: {img_filename}")
                    shutil.copy2(src_path, dst_path)
                else:
                    print(f"Warning: Image file not found: {src_path}")
                    continue
                
                merged_data['images'].append(img)
                
                # Update annotations for this image
                for ann in data['annotations']:
                    if ann['image_id'] == old_id:
                        ann['image_id'] = new_id
                        ann['id'] += max_annotation_id
                        merged_data['annotations'].append(ann)
            
            # Update max IDs
            if data['images']:
                max_image_id = max(img['id'] for img in merged_data['images'])
            if merged_data['annotations']:
                max_annotation_id = max(ann['id'] for ann in merged_data['annotations'])
            
            print(f"Processed {len(data['images'])} images and {len(data['annotations'])} annotations")
            
            # Cleanup temporary directory
            shutil.rmtree(temp_dir)
    
    # Save merged COCO JSON
    output_json = os.path.join(output_dir, 'merged_coco.json')
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\nMerged dataset statistics:")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"COCO JSON file: {output_json}")

if __name__ == "__main__":
    # Example usage
    dataset1_path = "dataset1.zip"  # Your dataset
    dataset2_path = "dataset2.zip"  # Abenathi's dataset
    output_dir = "merged_dataset"
    
    merge_coco_datasets(dataset1_path, dataset2_path, output_dir)