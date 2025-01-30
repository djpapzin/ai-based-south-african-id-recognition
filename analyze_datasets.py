import json
import os
from pathlib import Path
import hashlib

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read the file in chunks
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def analyze_dataset(dataset_dir):
    """Analyze a dataset for duplicates and check annotations"""
    print(f"\nAnalyzing dataset: {dataset_dir}")
    
    # Check directory structure
    images_dir = os.path.join(dataset_dir, "images")
    json_path = os.path.join(dataset_dir, "result.json")
    
    if not os.path.exists(images_dir):
        print("❌ Images directory not found!")
        return
    
    if not os.path.exists(json_path):
        print("❌ result.json not found!")
        return
    
    # Analyze images
    image_files = os.listdir(images_dir)
    print(f"\nImages:")
    print(f"✓ Total images found: {len(image_files)}")
    
    # Check for duplicate images by content (using hash)
    print("\nChecking for duplicate images...")
    image_hashes = {}
    for img in image_files:
        img_path = os.path.join(images_dir, img)
        if os.path.isfile(img_path):
            file_hash = calculate_file_hash(img_path)
            if file_hash in image_hashes:
                print(f"❌ Duplicate content found:")
                print(f"  - {img}")
                print(f"  - {image_hashes[file_hash]}")
            else:
                image_hashes[file_hash] = img
    
    # Check for duplicate filenames
    filename_counts = {}
    for img in image_files:
        if img in filename_counts:
            filename_counts[img] += 1
        else:
            filename_counts[img] = 1
    
    duplicate_filenames = {k: v for k, v in filename_counts.items() if v > 1}
    if duplicate_filenames:
        print("\n❌ Duplicate filenames found:")
        for filename, count in duplicate_filenames.items():
            print(f"  - {filename} (appears {count} times)")
    else:
        print("✓ No duplicate filenames found")
    
    # Analyze annotations
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\nAnnotations analysis:")
        print(f"✓ Images in JSON: {len(data.get('images', []))}")
        print(f"✓ Categories: {len(data.get('categories', []))}")
        
        annotations = data.get('annotations', [])
        if annotations:
            print(f"✓ Total annotations: {len(annotations)}")
            # Count annotations per image
            annotations_per_image = {}
            for ann in annotations:
                img_id = ann.get('image_id')
                if img_id is not None:
                    annotations_per_image[img_id] = annotations_per_image.get(img_id, 0) + 1
            
            images_with_annotations = len(annotations_per_image)
            print(f"✓ Images with annotations: {images_with_annotations}")
            print(f"✓ Images without annotations: {len(data['images']) - images_with_annotations}")
        else:
            print("⚠️ No annotations found in the dataset")
        
        # List categories
        print("\nCategories:")
        for cat in data.get('categories', []):
            print(f"  - {cat.get('name')} (id: {cat.get('id')})")
            
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in result.json")
    except Exception as e:
        print(f"❌ Error analyzing annotations: {str(e)}")

if __name__ == "__main__":
    # Analyze both datasets
    analyze_dataset("abenathi_object_detection_dataset")
    analyze_dataset("dj_object_detection_dataset") 