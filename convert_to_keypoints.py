import json
import os
from pathlib import Path

def convert_corners_to_keypoints(input_json, output_json):
    """Convert corner annotations to keypoint format."""
    print(f"Converting {input_json} to keypoint format...")
    
    # Load annotations
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Create a copy of the data
    new_data = {
        "images": data["images"],
        "categories": data["categories"],
        "annotations": []
    }
    
    # Process annotations by image
    image_anns = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_anns:
            image_anns[img_id] = []
        image_anns[img_id].append(ann)
    
    # Convert annotations for each image
    next_ann_id = 1
    for img_id, anns in image_anns.items():
        # Get document annotation
        doc_anns = [ann for ann in anns if ann["category_id"] == 6]  # id_document
        corner_anns = [ann for ann in anns if ann["category_id"] in [0, 1, 13, 14]]  # corners
        other_anns = [ann for ann in anns if ann["category_id"] not in [0, 1, 6, 13, 14]]  # other fields
        
        # If we have corners, create keypoint annotation
        if corner_anns:
            # Initialize keypoints array [x1,y1,v1,x2,y2,v2,x3,y3,v3,x4,y4,v4]
            keypoints = [0] * 12
            
            # Map corner types to indices
            corner_map = {
                0: 9,   # bottom_left -> 3
                1: 6,   # bottom_right -> 2
                13: 0,  # top_left -> 0
                14: 3   # top_right -> 1
            }
            
            # Fill keypoints array
            for corner in corner_anns:
                idx = corner_map.get(corner["category_id"])
                if idx is not None:
                    bbox = corner["bbox"]
                    x, y = int(bbox[0]), int(bbox[1])
                    keypoints[idx:idx+3] = [x, y, 2]  # 2 means visible
            
            # Create document annotation with keypoints
            doc_bbox = doc_anns[0]["bbox"] if doc_anns else [0, 0, 1000, 1000]
            keypoint_ann = {
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": 6,  # id_document
                "bbox": doc_bbox,
                "keypoints": keypoints,
                "num_keypoints": 4,
                "area": doc_bbox[2] * doc_bbox[3],
                "iscrowd": 0
            }
            new_data["annotations"].append(keypoint_ann)
            next_ann_id += 1
        
        # Add other field annotations
        for ann in other_anns:
            ann["id"] = next_ann_id
            new_data["annotations"].append(ann)
            next_ann_id += 1
    
    # Save converted annotations
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"Saved converted annotations to {output_json}")
    print(f"Total annotations: {len(new_data['annotations'])}")

def main():
    # Convert both train and val sets
    base_dir = "merged_dataset"
    for split in ['train', 'val']:
        input_json = os.path.join(base_dir, split, "annotations.json")
        output_json = os.path.join(base_dir, split, "annotations_keypoints.json")
        convert_corners_to_keypoints(input_json, output_json)

if __name__ == "__main__":
    main() 