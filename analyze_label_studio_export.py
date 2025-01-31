import json
import os
from collections import defaultdict
from pprint import pprint
import sys

def analyze_label_studio_export(json_path):
    """Analyze Label Studio JSON export and show keypoint structure."""
    print(f"\nAnalyzing Label Studio export: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {json_path}")
        return None
    
    print(f"\nTotal annotations: {len(data)}")
    
    # Find annotations with keypoints
    keypoint_annotations = []
    for item in data:
        if 'annotations' in item and item['annotations']:
            for annotation in item['annotations']:
                for result in annotation.get('result', []):
                    if 'type' in result and result['type'] == 'keypointlabels':
                        keypoint_annotations.append(result)
    
    print(f"\nFound {len(keypoint_annotations)} keypoint annotations")
    
    if keypoint_annotations:
        print("\nSample keypoint annotation structure:")
        pprint(keypoint_annotations[0])
        
        # Analyze keypoint types
        keypoint_types = defaultdict(int)
        for ann in keypoint_annotations:
            if 'value' in ann and 'labels' in ann['value']:
                for label in ann['value']['labels']:
                    keypoint_types[label] += 1
        
        print("\nKeypoint types and counts:")
        for kp_type, count in sorted(keypoint_types.items()):
            print(f"- {kp_type}: {count} annotations")

def convert_to_detectron2_coco(input_path, output_path=None):
    print(f"\nVerifying file: {input_path}")
    
    # Load Label Studio JSON
    with open(input_path) as f:
        data = json.load(f)
    
    # Print first item structure for debugging
    print("\nFirst item structure:")
    print(json.dumps(data[0] if isinstance(data, list) else data, indent=2))
    
    # Verify expected fields
    fields = list(data.keys()) if not isinstance(data, list) else list(data[0].keys())
    print("\nFound fields:", fields)
    expected_fields = ["annotations", "data", "id"]
    print("Expected fields:", expected_fields)
    
    # Print sample annotation structure
    if not isinstance(data, list) and "annotations" in data:
        print("\nSample annotation structure:")
        print(json.dumps(data["annotations"][0], indent=1))
    
    # If output path not specified, create one in same directory as input
    if output_path is None:
        input_dir = os.path.dirname(input_path)
        output_path = os.path.join(input_dir, "detectron2_coco.json")
    
    print(f"\nConverting {input_path} to Detectron2 COCO format...")
    
    # Initialize COCO format
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "citizenship_status"},
            {"id": 1, "name": "country_of_birth"},
            {"id": 2, "name": "date_of_birth"},
            {"id": 3, "name": "face"},
            {"id": 4, "name": "id_document"},
            {"id": 5, "name": "id_number"},
            {"id": 6, "name": "names"},
            {"id": 7, "name": "surname"},
            {"id": 8, "name": "corners"},
            {"id": 9, "name": "sex"},
            {"id": 10, "name": "nationality"},
            {"id": 11, "name": "signature"}
        ]
    }
    
    # Process each task
    annotation_id = 0
    for task in data if isinstance(data, list) else [data]:
        image_id = task["id"]
        file_name = task["file_upload"]
        
        # Add image
        coco_data["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": 400,  # Fixed size for now
            "height": 888
        })
        
        # Process annotations
        for ann in task["annotations"]:
            for result in ann["result"]:
                if result["type"] in ["rectanglelabels", "keypointlabels"]:
                    value = result["value"]
                    
                    # Handle bounding boxes
                    if result["type"] == "rectanglelabels":
                        category_name = value["rectanglelabels"][0]
                        try:
                            category_id = next(cat["id"] for cat in coco_data["categories"] if cat["name"] == category_name)
                        except StopIteration:
                            print(f"Warning: Skipping annotation with unknown category '{category_name}'")
                            continue
                        
                        # Convert relative coordinates to absolute
                        x = value["x"] * result["original_width"] / 100
                        y = value["y"] * result["original_height"] / 100
                        w = value["width"] * result["original_width"] / 100
                        h = value["height"] * result["original_height"] / 100
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                    
                    # Handle keypoints
                    elif result["type"] == "keypointlabels":
                        category_id = 8  # corners category
                        keypoint_name = value["keypointlabels"][0]
                        
                        # Convert relative coordinates to absolute
                        x = value["x"] * result["original_width"] / 100
                        y = value["y"] * result["original_height"] / 100
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "keypoints": [x, y, 2],  # 2 means visible keypoint
                            "num_keypoints": 1
                        })
                        annotation_id += 1
    
    # Save COCO format JSON
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete. Saved to: {output_path}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Total categories: {len(coco_data['categories'])}")
    
    print("\nCategories:")
    for cat in coco_data["categories"]:
        print(f"- {cat['name']} (id: {cat['id']})")

def verify_label_studio_format(json_path):
    """Verify if the JSON file is in the correct Label Studio export format."""
    print(f"\nVerifying file: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {json_path}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON file at {json_path}")
        return False
    
    # Check if it's a list (Label Studio JSON format) or dict (COCO format)
    if not isinstance(data, list):
        print("This appears to be in COCO format, not Label Studio JSON format.")
        print("Expected a list of annotations, but got a dictionary.")
        if isinstance(data, dict):
            print("\nFound top-level keys:", sorted(data.keys()))
        return False
    
    # Check first item structure
    if len(data) > 0:
        first_item = data[0]
        print("\nFirst item structure:")
        pprint(first_item)
        
        # Check for typical Label Studio JSON fields
        expected_fields = {'id', 'annotations', 'data'}
        found_fields = set(first_item.keys())
        
        print("\nFound fields:", sorted(found_fields))
        print("Expected fields:", sorted(expected_fields))
        
        if 'annotations' in first_item:
            print("\nSample annotation structure:")
            if first_item['annotations']:
                pprint(first_item['annotations'][0])
            else:
                print("No annotations found in first item")
    
    return True

if __name__ == "__main__":
    # Get input path from command line or use default
    input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("abenathi_object_detection_dataset", "abenathi_object_detection.json")
    
    # Get output path from command line or use default
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Verify Abenathi's export
    json_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\abenathi_object_detection.json"
    verify_label_studio_format(json_path)

    # Convert Abenathi's dataset
    convert_to_detectron2_coco(input_path, output_path)
