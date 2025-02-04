import json
from pathlib import Path

def convert_to_coco_keypoints():
    # Paths
    dataset_dir = Path(r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset")
    min_json_file = dataset_dir / "result_min.json"
    output_file = dataset_dir / "result_with_keypoints.json"
    
    # Load Label Studio export
    with open(min_json_file) as f:
        data = json.load(f)
    
    # Initialize COCO format
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "id_document", "supercategory": "document",
             "keypoints": ['top_left', 'top_right', 'bottom_right', 'bottom_left'],
             "skeleton": [[0, 1], [1, 2], [2, 3], [3, 0]]},
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
    annotation_id = 1
    for image_id, item in enumerate(data, 1):
        # Add image
        image_filename = Path(item["image"]).name
        image_path = f"images/{image_filename}"  # Use forward slashes for consistency
        
        # Get image dimensions
        width = item.get("original_width", 0)
        height = item.get("original_height", 0)
        
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,  # Just use the filename
            "width": width,
            "height": height
        })
        
        # Process keypoints for id_document
        if "corners" in item:
            corners = item["corners"]
            # Sort corners
            sorted_corners = sorted(corners, key=lambda x: (x["y"], x["x"]))
            top_corners = sorted(sorted_corners[:2], key=lambda x: x["x"])
            bottom_corners = sorted(sorted_corners[2:], key=lambda x: x["x"])
            ordered_corners = top_corners + list(reversed(bottom_corners))
            
            # Convert keypoints
            keypoints = []
            for corner in ordered_corners:
                x = corner.get("x", 0) * width / 100
                y = corner.get("y", 0) * height / 100
                keypoints.extend([x, y, 2])  # 2 means visible
            
            # Calculate bounding box from keypoints
            x_coords = [kp for i, kp in enumerate(keypoints) if i % 3 == 0]
            y_coords = [kp for i, kp in enumerate(keypoints) if i % 3 == 1]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add id_document annotation with keypoints
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "segmentation": [],
                "keypoints": keypoints,
                "num_keypoints": 4,
                "iscrowd": 0
            })
            annotation_id += 1
        elif "bbox" in item:
            # If no corners but bbox exists, look for id_document bbox
            for bbox in item["bbox"]:
                if "labels" in bbox and len(bbox["labels"]) > 0:
                    label = bbox["labels"][0]["value"]
                    if label == "id_document":
                        x = bbox["x"] * width / 100
                        y = bbox["y"] * height / 100
                        w = bbox["width"] * width / 100
                        h = bbox["height"] * height / 100
                        
                        # Generate keypoints from bbox corners
                        keypoints = [
                            x, y, 2,  # top-left
                            x + w, y, 2,  # top-right
                            x + w, y + h, 2,  # bottom-right
                            x, y + h, 2  # bottom-left
                        ]
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "segmentation": [],
                            "keypoints": keypoints,
                            "num_keypoints": 4,
                            "iscrowd": 0
                        })
                        annotation_id += 1
        
        # Process bounding boxes for fields
        if "bbox" in item:
            for bbox in item["bbox"]:
                if "labels" in bbox and len(bbox["labels"]) > 0:
                    label = bbox["labels"][0]["value"]
                    if label in category_map and label != "id_document":  # Skip id_document as it's handled above
                        x = bbox["x"] * width / 100
                        y = bbox["y"] * height / 100
                        w = bbox["width"] * width / 100
                        h = bbox["height"] * height / 100
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_map[label],
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1
        elif 'annotations' in item:
            for result in item['annotations']:
                if result['type'] == 'rectanglelabels':
                    # Convert percentage to absolute coordinates
                    x = result['value']['x'] * width / 100
                    y = result['value']['y'] * height / 100
                    w = result['value']['width'] * width / 100
                    h = result['value']['height'] * height / 100
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_map[result['value']['rectanglelabels'][0]],
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    annotation_id += 1
    
    # Save the COCO format data
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConverted annotations saved to: {output_file}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print("\nCategories:")
    for cat in coco_data["categories"]:
        print(f"  - {cat['name']} (ID: {cat['id']})")

if __name__ == "__main__":
    convert_to_coco_keypoints()
