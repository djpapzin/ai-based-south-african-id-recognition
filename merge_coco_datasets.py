import json
import os
import shutil
import sys

def merge_datasets(dataset1_path, dataset2_path, output_path):
    # Load datasets
    datasets = []
    for path in [dataset1_path, dataset2_path]:
        if os.path.exists(path):
            with open(path) as f:
                datasets.append(json.load(f))
        else:
            print(f"Warning: {path} not found")
            return
    
    # Merge categories from both datasets
    category_map = {}  # Map (dataset_idx, old_category_id) to new category ID
    merged_categories = []
    seen_categories = set()
    
    for dataset_idx, dataset in enumerate(datasets):
        for cat in dataset["categories"]:
            if cat["name"] not in seen_categories:
                new_id = len(merged_categories)
                category_map[(dataset_idx, cat["id"])] = new_id
                merged_categories.append({"id": new_id, "name": cat["name"]})
                seen_categories.add(cat["name"])
            else:
                # Find existing category ID
                existing_id = next(c["id"] for c in merged_categories if c["name"] == cat["name"])
                category_map[(dataset_idx, cat["id"])] = existing_id
    
    # Initialize merged dataset
    merged_data = {
        "images": [],
        "annotations": [],
        "categories": merged_categories
    }
    
    # Track max IDs
    max_image_id = 0
    max_annotation_id = 0
    
    # Process each dataset
    for i, data in enumerate(datasets):
        print(f"\nProcessing dataset {i+1}...")
        
        # Process images
        for img in data["images"]:
            old_id = img["id"]
            new_id = old_id + max_image_id
            img["id"] = new_id
            merged_data["images"].append(img)
            
            # Update corresponding annotations
            for ann in data["annotations"]:
                if ann["image_id"] == old_id:
                    ann["image_id"] = new_id
                    ann["id"] += max_annotation_id
                    # Update category ID
                    old_category_id = ann["category_id"]
                    ann["category_id"] = category_map[(i, old_category_id)]
                    merged_data["annotations"].append(ann)
        
        # Update max IDs for next dataset
        if data["images"]:
            max_image_id = max(img["id"] for img in merged_data["images"]) + 1
        if merged_data["annotations"]:
            max_annotation_id = max(ann["id"] for ann in merged_data["annotations"]) + 1
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save merged dataset
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"\nMerged dataset created at {output_path}:")
    print(f"Images: {len(merged_data['images'])}")
    print(f"Annotations: {len(merged_data['annotations'])}")
    print(f"Categories: {len(merged_data['categories'])}")
    print("\nCategories:")
    for cat in merged_data["categories"]:
        print(f"- {cat['name']} (id: {cat['id']})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merge_coco_datasets.py <dataset1_path> <dataset2_path> <output_path>")
        sys.exit(1)
        
    merge_datasets(sys.argv[1], sys.argv[2], sys.argv[3])