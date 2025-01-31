import json
import os
from pathlib import Path
import imghdr

def verify_dataset(merged_dir="merged_object_detection_dataset"):
    print("Verifying merged dataset integrity...")
    
    # Check directory structure
    images_dir = os.path.join(merged_dir, "images")
    json_path = os.path.join(merged_dir, "result.json")
    
    if not os.path.exists(merged_dir):
        print("❌ Error: Merged dataset directory not found!")
        return False
    
    if not os.path.exists(images_dir):
        print("❌ Error: Images directory not found!")
        return False
    
    if not os.path.exists(json_path):
        print("❌ Error: result.json not found!")
        return False
    
    # Load annotations
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"✓ Successfully loaded annotations file")
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON format in result.json")
        return False
    
    # Verify images
    image_files = set(os.listdir(images_dir))
    print(f"✓ Found {len(image_files)} images in images directory")
    
    # Check image integrity and prefixes
    abenathi_count = 0
    dj_count = 0
    corrupt_images = []
    
    for img in image_files:
        img_path = os.path.join(images_dir, img)
        # Check if it's a valid image
        if not imghdr.what(img_path):
            corrupt_images.append(img)
        
        # Count images by prefix
        if img.startswith('abenathi_'):
            abenathi_count += 1
        elif img.startswith('dj_'):
            dj_count += 1
    
    if corrupt_images:
        print(f"❌ Found {len(corrupt_images)} corrupt images:")
        for img in corrupt_images:
            print(f"  - {img}")
    else:
        print("✓ All images are valid")
    
    print(f"✓ Found {abenathi_count} images from Abenathi's dataset")
    print(f"✓ Found {dj_count} images from DJ's dataset")
    
    # Verify annotations
    referenced_images = set()
    invalid_annotations = []
    
    for i, ann in enumerate(annotations):
        if not isinstance(ann, dict):
            print(f"❌ Invalid annotation format at index {i}")
            continue
            
        if 'image' not in ann:
            invalid_annotations.append(f"Missing 'image' field at index {i}")
            continue
            
        image_name = os.path.basename(ann['image'])
        referenced_images.add(image_name)
        
        if image_name not in image_files:
            invalid_annotations.append(f"Referenced image not found: {image_name}")
    
    # Report findings
    print("\nAnnotation Statistics:")
    print(f"✓ Total annotations: {len(annotations)}")
    print(f"✓ Unique images referenced in annotations: {len(referenced_images)}")
    
    # Check for unreferenced images
    unreferenced_images = image_files - referenced_images
    if unreferenced_images:
        print(f"\n⚠️ Warning: Found {len(unreferenced_images)} images without annotations:")
        for img in sorted(unreferenced_images):
            print(f"  - {img}")
    
    # Report any invalid annotations
    if invalid_annotations:
        print(f"\n❌ Found {len(invalid_annotations)} invalid annotations:")
        for error in invalid_annotations:
            print(f"  - {error}")
    
    print("\nVerification Summary:")
    print(f"✓ Total images: {len(image_files)}")
    print(f"  - Abenathi's images: {abenathi_count}")
    print(f"  - DJ's images: {dj_count}")
    print(f"✓ Total annotations: {len(annotations)}")
    print(f"✓ Images referenced in annotations: {len(referenced_images)}")
    
    if not invalid_annotations and not corrupt_images:
        print("\n✅ Dataset integrity check passed!")
        return True
    else:
        print("\n⚠️ Dataset has some issues that need attention")
        return False

if __name__ == "__main__":
    verify_dataset() 