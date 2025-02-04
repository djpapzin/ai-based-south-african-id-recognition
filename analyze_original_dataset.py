import os
import json
from pathlib import Path
from collections import defaultdict

def analyze_dataset():
    """Analyze original and merged datasets to find annotation discrepancies"""
    # Define paths
    project_dir = Path(__file__).parent
    original_dir = project_dir / "dj_object_detection_dataset"
    merged_dir = project_dir / "merged_dataset"
    
    print("\nAnalyzing Original Dataset:")
    print("-" * 50)
    
    # Analyze original dataset structure
    if not original_dir.exists():
        print(f"Error: Original dataset directory not found at {original_dir}")
        return
        
    # Count all JSON files (potential annotation files)
    json_files = list(original_dir.glob("**/*.json"))
    print(f"\nFound {len(json_files)} JSON files in original dataset:")
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
                if 'images' in data and 'annotations' in data:
                    print(f"\nAnalyzing {json_file.relative_to(project_dir)}:")
                    print(f" - Images registered: {len(data['images'])}")
                    print(f" - Total annotations: {len(data['annotations'])}")
                    
                    # Count annotations per category
                    cat_counts = defaultdict(int)
                    for ann in data['annotations']:
                        cat_id = ann['category_id']
                        if 'categories' in data:
                            cat_name = next((cat['name'] for cat in data['categories'] if cat['id'] == cat_id), f"Unknown-{cat_id}")
                        else:
                            cat_name = f"Category-{cat_id}"
                        cat_counts[cat_name] += 1
                    
                    print("\nAnnotations per category:")
                    for cat, count in sorted(cat_counts.items()):
                        print(f" - {cat}: {count}")
        except Exception as e:
            print(f"Error reading {json_file}: {str(e)}")
    
    # Count all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in original_dir.glob("**/*") 
                  if f.suffix.lower() in image_extensions]
    
    print(f"\nFound {len(image_files)} images in original dataset")
    
    # Compare with merged dataset
    print("\nComparing with Merged Dataset:")
    print("-" * 50)
    
    # Analyze merged dataset
    train_ann = merged_dir / "train" / "annotations.json"
    val_ann = merged_dir / "val" / "annotations.json"
    
    def analyze_split(ann_path, split_name):
        if not ann_path.exists():
            print(f"{split_name} annotations not found at {ann_path}")
            return
            
        with open(ann_path) as f:
            data = json.load(f)
            print(f"\n{split_name} split:")
            print(f" - Images registered: {len(data['images'])}")
            print(f" - Total annotations: {len(data['annotations'])}")
            
            # Count annotations per category
            cat_counts = defaultdict(int)
            for ann in data['annotations']:
                cat_id = ann['category_id']
                if 'categories' in data:
                    cat_name = next((cat['name'] for cat in data['categories'] if cat['id'] == cat_id), f"Unknown-{cat_id}")
                else:
                    cat_name = f"Category-{cat_id}"
                cat_counts[cat_name] += 1
            
            print("\nAnnotations per category:")
            for cat, count in sorted(cat_counts.items()):
                print(f" - {cat}: {count}")
            
            return data['images']
    
    train_images = analyze_split(train_ann, "Training")
    val_images = analyze_split(val_ann, "Validation")
    
    # Find missing annotations
    if train_images and val_images:
        merged_image_names = set(img['file_name'].split('/')[-1] 
                               for img in train_images + val_images)
        original_image_names = set(f.name for f in image_files)
        
        missing = original_image_names - merged_image_names
        if missing:
            print("\nImages in original dataset but not in merged dataset:")
            for img in sorted(missing)[:5]:
                print(f" - {img}")
            if len(missing) > 5:
                print(f" ...and {len(missing)-5} more")

if __name__ == "__main__":
    print("Starting dataset analysis...")
    analyze_dataset()
