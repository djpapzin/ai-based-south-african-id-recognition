import json
import os

def analyze_dataset(json_path):
    print(f"\nAnalyzing {os.path.basename(json_path)}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Number of images: {len(data['images'])}")
    print(f"Number of annotations: {len(data['annotations'])}")
    print(f"Categories: {[cat['name'] for cat in data['categories']]}")
    
    # Check for potential issues
    print("\nChecking for issues:")
    
    # 1. Check images
    for img in data['images']:
        if not all(key in img for key in ['file_name', 'width', 'height', 'id']):
            print(f"❌ Image missing required fields: {img.get('file_name', 'unknown')}")
    
    # 2. Check annotations
    for ann in data['annotations']:
        if not all(key in ann for key in ['bbox', 'category_id', 'image_id']):
            print(f"❌ Annotation {ann.get('id', 'unknown')} missing required fields")
        if 'bbox' in ann and (len(ann['bbox']) != 4 or not all(isinstance(x, (int, float)) for x in ann['bbox'])):
            print(f"❌ Invalid bbox in annotation {ann.get('id', 'unknown')}: {ann.get('bbox', 'unknown')}")

# Analyze both datasets
base_dir = os.path.dirname(__file__)
abenathi_json = os.path.join(base_dir, 'abenathi_object_detection_dataset', 'detectron2_coco.json')
dj_json = os.path.join(base_dir, 'dj_object_detection_dataset', 'detectron2_coco.json')

analyze_dataset(abenathi_json)
analyze_dataset(dj_json)
