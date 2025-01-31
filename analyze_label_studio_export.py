import json
import os
from collections import defaultdict
from pprint import pprint

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

def convert_to_detectron2_coco(label_studio_path, output_path):
    """Convert Label Studio JSON to Detectron2-compatible COCO format with keypoint support."""
    print(f"\nConverting {label_studio_path} to Detectron2 COCO format...")
    
    with open(label_studio_path, 'r') as f:
        data = json.load(f)
    
    # Initialize COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define keypoint names in order (this order must be consistent)
    keypoint_names = ['top_left_corner', 'top_right_corner', 'bottom_left_corner', 'bottom_right_corner']
    
    # Create categories
    bbox_categories = set()
    for item in data:
        if 'annotations' in item and item['annotations']:
            for annotation in item['annotations']:
                for result in annotation.get('result', []):
                    if result['type'] == 'rectanglelabels':
                        bbox_categories.update(result['value']['rectanglelabels'])
    
    # Add bounding box categories
    category_id = 0
    category_map = {}
    for category in sorted(bbox_categories):
        coco_format["categories"].append({
            "id": category_id,
            "name": category,
            "supercategory": "none"
        })
        category_map[category] = category_id
        category_id += 1
    
    # Add keypoint category with all corner points
    coco_format["categories"].append({
        "id": category_id,
        "name": "corners",
        "supercategory": "none",
        "keypoints": keypoint_names,
        "skeleton": []  # Define if needed
    })
    keypoint_category_id = category_id
    
    # Process images and annotations
    image_id = 0
    ann_id = 0
    
    for item in data:
        if 'annotations' in item and item['annotations']:
            # Get image info
            image_filename = os.path.basename(item['data']['image'])
            original_height = None
            original_width = None
            
            # Get dimensions from first annotation result
            if item['annotations'][0]['result']:
                first_result = item['annotations'][0]['result'][0]
                original_height = first_result['original_height']
                original_width = first_result['original_width']
            
            # Add image
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_filename,
                "width": original_width,
                "height": original_height
            })
            
            # Process annotations
            for annotation in item['annotations']:
                # Collect keypoints for this image
                keypoints = {name: None for name in keypoint_names}
                
                for result in annotation.get('result', []):
                    # Handle bounding boxes
                    if result['type'] == 'rectanglelabels':
                        value = result['value']
                        x = value['x'] * original_width / 100  # Convert from percentage to pixels
                        y = value['y'] * original_height / 100
                        width = value['width'] * original_width / 100
                        height = value['height'] * original_height / 100
                        
                        for label in value['rectanglelabels']:
                            coco_format["annotations"].append({
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": category_map[label],
                                "bbox": [x, y, width, height],
                                "area": width * height,
                                "segmentation": [],
                                "iscrowd": 0
                            })
                            ann_id += 1
                    
                    # Collect keypoints
                    elif result['type'] == 'keypointlabels':
                        value = result['value']
                        x = value['x'] * original_width / 100
                        y = value['y'] * original_height / 100
                        keypoint_label = value['keypointlabels'][0]
                        keypoints[keypoint_label] = (x, y)
                
                # Add keypoint annotation if we found any keypoints
                if any(kp is not None for kp in keypoints.values()):
                    # Convert keypoints to COCO format [x1,y1,v1,x2,y2,v2,...]
                    kp_list = []
                    for name in keypoint_names:
                        if keypoints[name] is not None:
                            kp_list.extend([keypoints[name][0], keypoints[name][1], 2])
                        else:
                            kp_list.extend([0, 0, 0])
                    
                    # Calculate bounding box that encompasses all keypoints
                    valid_points = [(x, y) for x, y, v in zip(kp_list[::3], kp_list[1::3], kp_list[2::3]) if v > 0]
                    if valid_points:
                        x_coords, y_coords = zip(*valid_points)
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        coco_format["annotations"].append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": keypoint_category_id,
                            "keypoints": kp_list,
                            "num_keypoints": sum(1 for v in kp_list[2::3] if v > 0),
                            "bbox": [x_min, y_min, width, height],
                            "area": width * height,
                            "iscrowd": 0
                        })
                        ann_id += 1
            
            image_id += 1
    
    # Save converted format
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"\nConversion complete. Saved to: {output_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")
    print(f"Total categories: {len(coco_format['categories'])}")
    print("\nCategories:")
    for cat in coco_format['categories']:
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
    # Verify Abenathi's export
    json_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\abenathi_object_detection.json"
    verify_label_studio_format(json_path)

    # Convert Abenathi's dataset
    input_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\abenathi_object_detection.json"
    output_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\detectron2_coco.json"
    convert_to_detectron2_coco(input_path, output_path)
