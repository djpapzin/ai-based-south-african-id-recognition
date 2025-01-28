import json
import os
import shutil
from zipfile import ZipFile

# Create output directory
os.makedirs("merged_dataset/images", exist_ok=True)

# Initialize merged data
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

# Process dataset1
print("Processing dataset1.zip...")
with ZipFile("dataset1.zip") as zf:
    zf.extractall("temp_dataset1")

with open("temp_dataset1/result.json") as f:
    data1 = json.load(f)

# Add images and annotations from dataset1
for img in data1["images"]:
    old_id = img["id"]
    img["id"] = old_id + max_image_id
    
    # Copy image
    src = os.path.join("temp_dataset1/images", os.path.basename(img["file_name"]))
    dst = os.path.join("merged_dataset/images", os.path.basename(img["file_name"]))
    if os.path.exists(src):
        shutil.copy2(src, dst)
        merged_data["images"].append(img)
        print(f"Copied image: {img['file_name']}")
    
    # Add annotations
    for ann in data1["annotations"]:
        if ann["image_id"] == old_id:
            ann["image_id"] = img["id"]
            ann["id"] += max_annotation_id
            merged_data["annotations"].append(ann)

# Update max IDs
if data1["images"]:
    max_image_id = max(img["id"] for img in merged_data["images"])
if merged_data["annotations"]:
    max_annotation_id = max(ann["id"] for ann in merged_data["annotations"])

# Process dataset2
print("\nProcessing dataset2.zip...")
with ZipFile("dataset2.zip") as zf:
    zf.extractall("temp_dataset2")

with open("temp_dataset2/result.json") as f:
    data2 = json.load(f)

# Add images and annotations from dataset2
for img in data2["images"]:
    old_id = img["id"]
    img["id"] = old_id + max_image_id
    
    # Copy image
    src = os.path.join("temp_dataset2/images", os.path.basename(img["file_name"]))
    dst = os.path.join("merged_dataset/images", os.path.basename(img["file_name"]))
    if os.path.exists(src):
        shutil.copy2(src, dst)
        merged_data["images"].append(img)
        print(f"Copied image: {img['file_name']}")
    
    # Add annotations
    for ann in data2["annotations"]:
        if ann["image_id"] == old_id:
            ann["image_id"] = img["id"]
            ann["id"] += max_annotation_id
            merged_data["annotations"].append(ann)

# Save merged dataset
with open("merged_dataset/merged_coco.json", "w") as f:
    json.dump(merged_data, f, indent=2)

# Print statistics
print(f"\nMerged dataset created:")
print(f"Images: {len(merged_data['images'])}")
print(f"Annotations: {len(merged_data['annotations'])}")

# Cleanup
shutil.rmtree("temp_dataset1")
shutil.rmtree("temp_dataset2") 