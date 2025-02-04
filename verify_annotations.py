import json
from pathlib import Path
import os

def verify_annotations():
    # Paths
    dataset_dir = Path(r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset")
    images_dir = dataset_dir / "images"
    min_json_file = dataset_dir / "result_min.json"
    full_json_file = dataset_dir / "result.json"
    
    # Get local images
    local_images = {f.name.lower() for f in images_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg']}
    print(f"\nLocal Images Directory:")
    print(f"Total images found: {len(local_images)}")
    
    # Load min annotations (direct format)
    with open(min_json_file) as f:
        min_annotations = json.load(f)
    min_annotated_images = {Path(ann['image']).name.lower() for ann in min_annotations}
    
    # Load full annotations (COCO format)
    with open(full_json_file) as f:
        full_data = json.load(f)
    full_annotated_images = {Path(img['file_name']).name.lower() for img in full_data['images']}
    
    print(f"\nAnnotations Files:")
    print(f"result_min.json annotations: {len(min_annotations)}")
    print(f"result.json images: {len(full_data['images'])}")
    print(f"result.json annotations: {len(full_data['annotations'])}")
    
    print(f"\nUnique Images in Annotations:")
    print(f"result_min.json unique images: {len(min_annotated_images)}")
    print(f"result.json unique images: {len(full_annotated_images)}")
    
    # Compare sets
    print(f"\nComparison:")
    print(f"Images in result.json but not in result_min.json: {len(full_annotated_images - min_annotated_images)}")
    print(f"Images in result_min.json but not in result.json: {len(min_annotated_images - full_annotated_images)}")
    
    # Compare with local files
    print(f"\nLocal File Comparison:")
    print(f"Images with annotations in result.json but missing locally: {len(full_annotated_images - local_images)}")
    print(f"Local images without annotations in result.json: {len(local_images - full_annotated_images)}")
    
    if full_annotated_images - local_images:
        print("\nAnnotated images missing from local directory:")
        for img in sorted(full_annotated_images - local_images):
            print(f"- {img}")
            
    if local_images - full_annotated_images:
        print("\nLocal images missing annotations:")
        for img in sorted(local_images - full_annotated_images):
            print(f"- {img}")

if __name__ == '__main__':
    verify_annotations()
