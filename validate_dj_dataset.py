import json
import os
from pathlib import Path

def get_images_in_dir(dir_path: str) -> set:
    """Get all JPEG images in a directory."""
    if not os.path.exists(dir_path):
        print(f"Warning: Directory {dir_path} does not exist")
        return set()
    return set(os.path.basename(str(p)) for p in Path(dir_path).glob("*.[jJ][pP][eE][gG]"))

def validate_dj_dataset(dataset_path: str):
    """
    Validate DJ's dataset for completeness and correctness.
    
    Args:
        dataset_path: Path to the dataset directory containing images/ and result.json
    """
    # Load annotations
    json_path = os.path.join(dataset_path, "result.json")
    if not os.path.exists(json_path):
        print(f"\n❌ Error: result.json not found in {dataset_path}")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all physical images
    image_dir = os.path.join(dataset_path, "images")
    physical_images = get_images_in_dir(image_dir)
    print(f"\n1. Physical Images Check:")
    print(f"   Found {len(physical_images)} physical images in {image_dir}")
    
    # Check images in JSON
    json_images = {img["file_name"].split("\\")[-1]: img for img in data["images"]}
    print(f"\n2. JSON Images Check:")
    print(f"   Found {len(json_images)} images defined in result.json")
    
    # Cross reference physical vs JSON images
    missing_in_json = physical_images - set(json_images.keys())
    missing_in_folder = set(json_images.keys()) - physical_images
    
    if missing_in_json:
        print("\n   ⚠️ Images in folder but missing in JSON:")
        for img in sorted(missing_in_json):
            print(f"   - {img}")
    
    if missing_in_folder:
        print("\n   ⚠️ Images in JSON but missing in folder:")
        for img in sorted(missing_in_folder):
            print(f"   - {img}")
    
    # Check annotations
    print(f"\n3. Annotations Check:")
    annotations = data["annotations"]
    print(f"   Total annotations: {len(annotations)}")
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in annotations:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Check images without annotations
    images_without_annotations = []
    for img_id, img_info in enumerate(data["images"]):
        if img_id not in annotations_by_image:
            images_without_annotations.append(img_info["file_name"])
    
    if images_without_annotations:
        print("\n   ⚠️ Images without annotations:")
        for img in sorted(images_without_annotations):
            print(f"   - {img}")
    
    # Validate bounding boxes
    print("\n4. Bounding Box Validation:")
    invalid_boxes = []
    for img_id, img_info in enumerate(data["images"]):
        if img_id in annotations_by_image:
            width = img_info["width"]
            height = img_info["height"]
            
            for ann in annotations_by_image[img_id]:
                bbox = ann["bbox"]
                # COCO format: [x, y, width, height]
                if len(bbox) != 4:
                    invalid_boxes.append((img_info["file_name"], "Invalid format"))
                    continue
                
                x, y, w, h = bbox
                
                # Check if box dimensions are positive
                if w <= 0 or h <= 0:
                    invalid_boxes.append((img_info["file_name"], "Non-positive dimensions"))
                    continue
                
                # Check if box is within image boundaries
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    invalid_boxes.append((img_info["file_name"], "Out of bounds"))
                    continue
    
    if invalid_boxes:
        print("\n   ⚠️ Invalid bounding boxes found:")
        for img, reason in invalid_boxes:
            print(f"   - {img}: {reason}")
    
    # Check categories
    print("\n5. Category Check:")
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    print(f"   Found {len(categories)} categories:")
    for cat_id, cat_name in categories.items():
        print(f"   - {cat_id}: {cat_name}")
    
    # Check for annotations with invalid categories
    invalid_categories = []
    for ann in annotations:
        if ann["category_id"] not in categories:
            invalid_categories.append((
                data["images"][ann["image_id"]]["file_name"],
                ann["category_id"]
            ))
    
    if invalid_categories:
        print("\n   ⚠️ Annotations with invalid category IDs:")
        for img, cat_id in invalid_categories:
            print(f"   - {img}: category_id {cat_id}")
            
    # Print annotation statistics by category
    print("\n6. Annotation Statistics by Category:")
    category_counts = {}
    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = categories.get(cat_id, f"Unknown category {cat_id}")
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    for cat_name, count in sorted(category_counts.items()):
        print(f"   - {cat_name}: {count} annotations")

if __name__ == "__main__":
    dataset_path = r"C:\Users\lfana\Downloads\dj_dataset_02feb"
    validate_dj_dataset(dataset_path) 