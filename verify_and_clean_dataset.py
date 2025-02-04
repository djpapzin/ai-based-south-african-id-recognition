import json
import os
from pathlib import Path
import shutil
import cv2

def fix_file_extensions(data, img_dir):
    """Fix file extensions in annotations to match actual files"""
    fixed_images = []
    for img in data['images']:
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(img['file_name']))[0]
        
        # Check for both .jpg and .jpeg
        jpg_path = img_dir / f"{base_name}.jpg"
        jpeg_path = img_dir / f"{base_name}.jpeg"
        
        if jpg_path.exists():
            img['file_name'] = f"{base_name}.jpg"
        elif jpeg_path.exists():
            img['file_name'] = f"{base_name}.jpeg"
            
        fixed_images.append(img)
    return fixed_images

def verify_dataset(base_dir="merged_dataset"):
    """Verify the dataset structure and annotations"""
    print("\n=== Dataset Verification Report ===")
    
    dataset_stats = {}
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        print("-" * 50)
        
        split_dir = Path(base_dir) / split
        img_dir = split_dir / "images"
        ann_file = split_dir / "annotations.json"
        
        print("1. Directory Structure:")
        print(f"- Images directory: {'✓' if img_dir.exists() else '✗'} ({img_dir})")
        print(f"- Annotation file: {'✓' if ann_file.exists() else '✗'} ({ann_file})")
        
        if not (img_dir.exists() and ann_file.exists()):
            print("ERROR: Missing required directories or files!")
            continue
            
        # Count actual image files
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg"))
        print(f"\n2. Image Files:")
        print(f"- Total images in directory: {len(images)}")
        
        # Load and fix annotations
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Fix file extensions
        data['images'] = fix_file_extensions(data, img_dir)
        
        print(f"\n3. Annotation File Content:")
        print(f"- Images in annotations: {len(data['images'])}")
        print(f"- Total annotations: {len(data['annotations'])}")
        print(f"- Categories: {len(data['categories'])}")
        
        # Check image-annotation matching
        img_ids = {img['id']: img['file_name'] for img in data['images']}
        ann_per_img = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            ann_per_img[img_id] = ann_per_img.get(img_id, 0) + 1
        
        print("\n4. Annotation Distribution:")
        print(f"- Images with annotations: {len(ann_per_img)}")
        print(f"- Images without annotations: {len(img_ids) - len(ann_per_img)}")
        
        # Verify image files exist
        missing_files = []
        for img in data['images']:
            img_path = img_dir / os.path.basename(img['file_name'])
            if not img_path.exists():
                missing_files.append(img['file_name'])
        
        if missing_files:
            print("\n5. Missing Image Files:")
            for f in missing_files[:5]:
                print(f"- {f}")
            if len(missing_files) > 5:
                print(f"... and {len(missing_files)-5} more")
        else:
            print("\n5. All referenced images exist: ✓")
        
        # Save fixed annotations
        with open(ann_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        dataset_stats[split] = {
            'images': len(images),
            'annotations': len(data['annotations']),
            'categories': data['categories']
        }
    
    return dataset_stats

def clean_dataset(base_dir="merged_dataset", output_dir="merged_dataset_clean"):
    """Create a clean version of the dataset with only annotated images"""
    print("\n=== Creating Clean Dataset ===")
    
    for split in ['train', 'val']:
        print(f"\nProcessing {split} set:")
        
        # Setup directories
        out_split_dir = Path(output_dir) / split
        out_img_dir = out_split_dir / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        ann_file = Path(base_dir) / split / "annotations.json"
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Create new annotation file
        new_data = {
            'categories': data['categories'],
            'images': [],
            'annotations': []
        }
        
        # Process images with annotations
        next_img_id = 1
        img_id_map = {}
        
        for img in data['images']:
            old_id = img['id']
            img_anns = [ann for ann in data['annotations'] if ann['image_id'] == old_id]
            
            if img_anns:
                src_img = Path(base_dir) / split / "images" / os.path.basename(img['file_name'])
                if src_img.exists():
                    dst_img = out_img_dir / src_img.name
                    shutil.copy2(src_img, dst_img)
                    
                    new_img = img.copy()
                    new_img['id'] = next_img_id
                    new_img['file_name'] = dst_img.name
                    new_data['images'].append(new_img)
                    
                    for ann in img_anns:
                        new_ann = ann.copy()
                        new_ann['image_id'] = next_img_id
                        new_data['annotations'].append(new_ann)
                    
                    img_id_map[old_id] = next_img_id
                    next_img_id += 1
        
        # Save new annotations
        out_ann_file = out_split_dir / "annotations.json"
        with open(out_ann_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"- Images processed: {len(new_data['images'])}")
        print(f"- Annotations saved: {len(new_data['annotations'])}")
        print(f"- Output directory: {out_split_dir}")

if __name__ == "__main__":
    # First verify and fix the current dataset
    stats = verify_dataset()
    
    # Ask user if they want to create a clean version
    response = input("\nWould you like to create a clean version of the dataset? (y/n): ")
    if response.lower() == 'y':
        clean_dataset()
        print("\nClean dataset created. Verifying clean dataset:")
        verify_dataset("merged_dataset_clean")