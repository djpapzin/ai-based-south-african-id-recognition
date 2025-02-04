import json
import os
from pathlib import Path
import random
from PIL import Image

def verify_image_and_get_dimensions(img_path):
    """Verify image and get its dimensions."""
    try:
        with Image.open(img_path) as img:
            img.verify()
            # Reopen to get size after verify
            img = Image.open(img_path)
            return img.size
    except Exception as e:
        print(f"Error verifying {img_path}: {str(e)}")
        return None

def fix_image_path(img_path):
    """Fix image path to get just the filename."""
    # Remove /data/upload prefix if present
    if '/data/upload' in img_path:
        img_path = img_path.split('/data/upload/')[-1]
    
    # Remove any numeric prefix and trailing path
    parts = img_path.split('/')
    filename = parts[-1]
    
    # Handle both .jpg and .jpeg extensions
    if filename.lower().endswith('.jpeg'):
        filename = filename[:-5] + '.jpg'
    elif not filename.lower().endswith('.jpg'):
        filename = filename + '.jpg'
    
    return filename

def convert_to_coco(input_json, output_json, image_dir, data_subset=None):
    """Convert Label Studio annotations to COCO format.
    
    Args:
        input_json: Path to input Label Studio JSON file
        output_json: Path to output COCO format JSON file
        image_dir: Path to directory containing images
        data_subset: List of image IDs to include in this conversion (for train/val split)
    """
    print(f"Converting {input_json} to COCO format...")
    
    # Load Label Studio annotations
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Filter data if subset is provided
    if data_subset is not None:
        data = [item for item in data if item["id"] in data_subset]
    
    # Initialize COCO format
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "id_document", "supercategory": "document"},
            {"id": 2, "name": "surname", "supercategory": "field"},
            {"id": 3, "name": "names", "supercategory": "field"},
            {"id": 4, "name": "sex", "supercategory": "field"},
            {"id": 5, "name": "nationality", "supercategory": "field"},
            {"id": 6, "name": "id_number", "supercategory": "field"},
            {"id": 7, "name": "date_of_birth", "supercategory": "field"},
            {"id": 8, "name": "country_of_birth", "supercategory": "field"},
            {"id": 9, "name": "citizenship_status", "supercategory": "field"},
            {"id": 10, "name": "face", "supercategory": "field"},
            {"id": 11, "name": "signature", "supercategory": "field"}
        ]
    }
    
    # Create category name to ID mapping
    category_map = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
    
    # Process each image
    next_ann_id = 1
    for img_id, img_data in enumerate(data, 1):
        # Get image filename and verify it exists
        filename = fix_image_path(img_data["image"])
        img_path = os.path.join(image_dir, filename)
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            print(f"Warning: Image not found {img_path}, skipping...")
            continue
        
        # Verify image and get dimensions
        dimensions = verify_image_and_get_dimensions(img_path)
        if dimensions is None:
            print(f"Warning: Invalid image {filename}, skipping...")
            continue
            
        width, height = dimensions
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        # Process bounding boxes
        for bbox in img_data["bbox"]:
            # Get category
            if "rectanglelabels" not in bbox or not bbox["rectanglelabels"]:
                continue
                
            category = bbox["rectanglelabels"][0]
            if category not in category_map:
                print(f"Warning: Unknown category {category}")
                continue
            
            # Convert relative coordinates to absolute
            x_rel = bbox["x"] / 100.0
            y_rel = bbox["y"] / 100.0
            w_rel = bbox["width"] / 100.0
            h_rel = bbox["height"] / 100.0
            
            x = x_rel * width
            y = y_rel * height
            w = w_rel * width
            h = h_rel * height
            
            # Add annotation
            coco_data["annotations"].append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": category_map[category],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            next_ann_id += 1
        
        # Process keypoints if present
        if "corners" in img_data:
            keypoints = []
            for corner in img_data["corners"]:
                x_rel = corner["x"] / 100.0
                y_rel = corner["y"] / 100.0
                x = x_rel * width
                y = y_rel * height
                keypoints.extend([x, y, 2])  # 2 means visible
            
            # Add keypoint annotation
            if len(keypoints) == 12:  # 4 corners * 3 values (x,y,v)
                coco_data["annotations"].append({
                    "id": next_ann_id,
                    "image_id": img_id,
                    "category_id": category_map["id_document"],
                    "keypoints": keypoints,
                    "num_keypoints": 4,
                    "bbox": [0, 0, width, height],  # Full image bbox for document
                    "area": width * height,
                    "iscrowd": 0,
                    "segmentation": []
                })
                next_ann_id += 1
    
    # Save COCO format annotations
    output_dir = os.path.dirname(output_json)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Conversion complete. Saved to {output_json}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    
    # Print annotation counts per category
    category_counts = {}
    for ann in coco_data["annotations"]:
        cat_id = ann["category_id"]
        cat_name = next(cat["name"] for cat in coco_data["categories"] if cat["id"] == cat_id)
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    print("\nAnnotations per category:")
    for cat_name, count in sorted(category_counts.items()):
        print(f"  - {cat_name}: {count}")

if __name__ == "__main__":
    # First verify and fix images
    from validate_annotations import verify_and_fix_images
    
    print("Verifying and fixing images...")
    verify_and_fix_images("merged_dataset/train/images")
    verify_and_fix_images("merged_dataset/val/images")
    
    # Load all data
    with open("dj_object_detection_dataset/result_min.json", 'r') as f:
        all_data = json.load(f)
    
    # Get all image IDs
    all_ids = [item["id"] for item in all_data]
    
    # Create train/val split (80/20)
    random.seed(42)  # For reproducibility
    val_size = int(len(all_ids) * 0.2)
    val_ids = set(random.sample(all_ids, val_size))
    train_ids = set(all_ids) - val_ids
    
    print(f"Total images: {len(all_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # Convert training set
    convert_to_coco(
        input_json="dj_object_detection_dataset/result_min.json",
        output_json="merged_dataset/train/annotations.json",
        image_dir="merged_dataset/train/images",
        data_subset=train_ids
    )
    
    # Convert validation set
    convert_to_coco(
        input_json="dj_object_detection_dataset/result_min.json",
        output_json="merged_dataset/val/annotations.json",
        image_dir="merged_dataset/val/images",
        data_subset=val_ids
    ) 