import os
import json
from pathlib import Path
import shutil
from datetime import datetime

def analyze_and_fix_annotations():
    """Analyze and fix annotation registration issues"""
    project_dir = Path(__file__).parent
    
    # Define paths
    train_dir = project_dir / "merged_dataset" / "train"
    val_dir = project_dir / "merged_dataset" / "val"
    train_ann_path = train_dir / "annotations.json"
    val_ann_path = val_dir / "annotations.json"
    
    def analyze_split(split_dir, ann_path, split_name):
        print(f"\nAnalyzing {split_name} split:")
        
        # Load annotations
        with open(ann_path) as f:
            annotations = json.load(f)
            
        # Get physical images
        image_dir = split_dir / "images"
        physical_images = set(f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        print(f"Physical images found: {len(physical_images)}")
        
        # Get registered images
        registered_images = set(img['file_name'].split('/')[-1] 
                              for img in annotations['images'])
        print(f"Registered images: {len(registered_images)}")
        
        # Find discrepancies
        unregistered = physical_images - registered_images
        missing_files = registered_images - physical_images
        
        if unregistered:
            print(f"\nUnregistered images ({len(unregistered)}):")
            for img in sorted(unregistered)[:5]:
                print(f" - {img}")
            if len(unregistered) > 5:
                print(f" ...and {len(unregistered)-5} more")
                
        if missing_files:
            print(f"\nRegistered but missing files ({len(missing_files)}):")
            for img in sorted(missing_files)[:5]:
                print(f" - {img}")
            if len(missing_files) > 5:
                print(f" ...and {len(missing_files)-5} more")
        
        return annotations, physical_images, registered_images

    def fix_annotations(annotations, physical_images, registered_images, split_name, ann_path):
        print(f"\nFixing {split_name} annotations:")
        
        # Backup original annotations
        backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = ann_path.parent / f"annotations_backup_{backup_time}.json"
        shutil.copy2(ann_path, backup_path)
        print(f"Created backup at: {backup_path}")
        
        # Add entries for unregistered images
        next_image_id = max(img['id'] for img in annotations['images']) + 1
        unregistered = physical_images - registered_images
        
        for img_file in unregistered:
            new_image = {
                'id': next_image_id,
                'file_name': f"images/{img_file}",
                'height': 0,  # Will be updated when training
                'width': 0    # Will be updated when training
            }
            annotations['images'].append(new_image)
            next_image_id += 1
            
        # Remove entries for missing files
        missing_files = registered_images - physical_images
        if missing_files:
            # Remove image entries
            annotations['images'] = [
                img for img in annotations['images']
                if img['file_name'].split('/')[-1] not in missing_files
            ]
            
            # Remove corresponding annotations
            valid_image_ids = set(img['id'] for img in annotations['images'])
            annotations['annotations'] = [
                ann for ann in annotations['annotations']
                if ann['image_id'] in valid_image_ids
            ]
        
        print(f"Added {len(unregistered)} new image entries")
        print(f"Removed {len(missing_files)} missing file entries")
        
        # Save updated annotations
        with open(ann_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved updated annotations to: {ann_path}")
        
        return annotations

    # Process training split
    train_ann, train_physical, train_registered = analyze_split(
        train_dir, train_ann_path, "training")
    train_ann = fix_annotations(
        train_ann, train_physical, train_registered, "training", train_ann_path)
    
    # Process validation split
    val_ann, val_physical, val_registered = analyze_split(
        val_dir, val_ann_path, "validation")
    val_ann = fix_annotations(
        val_ann, val_physical, val_registered, "validation", val_ann_path)
    
    print("\nAnnotation fixing complete!")

if __name__ == "__main__":
    print("Starting annotation analysis and fixing...")
    analyze_and_fix_annotations()
