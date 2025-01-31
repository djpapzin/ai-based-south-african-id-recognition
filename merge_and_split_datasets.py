import json
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

class DatasetMerger:
    def __init__(self, output_dir="merged_dataset"):
        self.output_dir = output_dir
        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "val")
        
        # Create output directories
        for dir_path in [self.train_dir, self.val_dir]:
            os.makedirs(os.path.join(dir_path, "images"), exist_ok=True)
    
    def load_json(self, json_path):
        """Load a COCO format JSON file."""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def merge_datasets(self, dataset1_path, dataset2_path):
        """Merge two COCO datasets."""
        print(f"Loading dataset 1 from {dataset1_path}")
        dataset1 = self.load_json(dataset1_path)
        print(f"Loading dataset 2 from {dataset2_path}")
        dataset2 = self.load_json(dataset2_path)
        
        # Initialize merged dataset
        merged = {
            'images': [],
            'annotations': [],
            'categories': dataset1['categories']  # Use categories from first dataset
        }
        
        # Track the maximum IDs to avoid conflicts
        max_image_id = 0
        max_annotation_id = 0
        
        # Helper function to add dataset
        def add_dataset(dataset, dataset_path, dataset_name):
            nonlocal max_image_id, max_annotation_id
            
            # Get the parent directory of the JSON file and find the images directory
            base_dir = os.path.dirname(dataset_path)
            image_dir = os.path.join(base_dir, "images")
            print(f"\nProcessing {dataset_name}:")
            print(f"Image directory: {image_dir}")
            
            # Create ID mapping for images
            old_to_new_image_ids = {}
            
            # Process images
            for img in dataset['images']:
                new_image_id = max_image_id + 1
                old_to_new_image_ids[img['id']] = new_image_id
                
                # Update image entry
                img_copy = img.copy()
                img_copy['id'] = new_image_id
                # Remove any 'images/' prefix and normalize path
                img_copy['file_name'] = os.path.basename(img['file_name'].replace('images/', ''))
                merged['images'].append(img_copy)
                
                # Check if image exists
                src_path = os.path.join(image_dir, img_copy['file_name'])
                if os.path.exists(src_path):
                    print(f"Found image: {img_copy['file_name']}")
                else:
                    print(f"Warning: Image not found: {src_path}")
                
                max_image_id = new_image_id
            
            # Process annotations
            for ann in dataset['annotations']:
                new_annotation_id = max_annotation_id + 1
                
                # Update annotation entry
                ann_copy = ann.copy()
                ann_copy['id'] = new_annotation_id
                ann_copy['image_id'] = old_to_new_image_ids[ann['image_id']]
                merged['annotations'].append(ann_copy)
                
                max_annotation_id = new_annotation_id
            
            return image_dir
        
        # Add both datasets
        print("\nMerging datasets...")
        image_dir1 = add_dataset(dataset1, dataset1_path, "Dataset 1 (Abenathi)")
        image_dir2 = add_dataset(dataset2, dataset2_path, "Dataset 2 (DJ)")
        
        return merged, [image_dir1, image_dir2]
    
    def copy_image(self, filename, source_dirs, target_dir):
        """Copy image from source directories to target directory."""
        # Ensure we only use the base filename
        clean_filename = os.path.basename(filename.replace('images/', ''))
        
        for source_dir in source_dirs:
            src_path = os.path.join(source_dir, clean_filename)
            if os.path.exists(src_path):
                dst_path = os.path.join(target_dir, clean_filename)
                print(f"Copying {clean_filename} to {target_dir}")
                shutil.copy2(src_path, dst_path)
                return True
        print(f"Warning: Image {clean_filename} not found in any source directory")
        return False

    def split_dataset(self, merged_data, image_dirs, val_size=0.2, random_state=42):
        """Split merged dataset into train and validation sets."""
        # Get all image IDs and filenames
        images_info = [(img['id'], img['file_name']) for img in merged_data['images']]
        
        # Split image info
        train_info, val_info = train_test_split(
            images_info,
            test_size=val_size,
            random_state=random_state
        )
        
        # Create train and validation datasets
        train_data = {
            'images': [],
            'annotations': [],
            'categories': merged_data['categories']
        }
        val_data = {
            'images': [],
            'annotations': [],
            'categories': merged_data['categories']
        }
        
        # Helper function to find image by ID
        def find_image(image_id):
            return next(img for img in merged_data['images'] if img['id'] == image_id)
        
        # Process train split
        train_image_ids = set()
        failed_copies = []
        print("\nProcessing training set:")
        for image_id, filename in train_info:
            train_image_ids.add(image_id)
            train_data['images'].append(find_image(image_id))
            success = self.copy_image(filename, image_dirs, os.path.join(self.train_dir, "images"))
            if not success:
                failed_copies.append(('train', filename))
        
        # Process validation split
        val_image_ids = set()
        print("\nProcessing validation set:")
        for image_id, filename in val_info:
            val_image_ids.add(image_id)
            val_data['images'].append(find_image(image_id))
            success = self.copy_image(filename, image_dirs, os.path.join(self.val_dir, "images"))
            if not success:
                failed_copies.append(('val', filename))
        
        # Add annotations to respective splits
        for ann in merged_data['annotations']:
            if ann['image_id'] in train_image_ids:
                train_data['annotations'].append(ann)
            elif ann['image_id'] in val_image_ids:
                val_data['annotations'].append(ann)
        
        # Report any failed copies
        if failed_copies:
            print("\nWarning: Failed to copy the following images:")
            for split, filename in failed_copies:
                print(f"- {filename} (intended for {split} set)")
        
        return train_data, val_data
    
    def save_dataset(self, data, output_path):
        """Save dataset to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    # Define paths
    abenathi_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\abenathi_object_detection_dataset\result.json"
    dj_path = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\dj_object_detection_dataset\result.json"
    output_dir = r"C:\Users\lfana\Documents\Kwantu\Machine Learning\merged_dataset"
    
    # Create merger instance
    merger = DatasetMerger(output_dir)
    
    # Merge datasets
    print("Merging datasets...")
    merged_data, image_dirs = merger.merge_datasets(abenathi_path, dj_path)
    
    # Split dataset
    print("\nSplitting into train/val sets...")
    train_data, val_data = merger.split_dataset(merged_data, image_dirs, val_size=0.2)
    
    # Save split datasets
    print("\nSaving datasets...")
    merger.save_dataset(train_data, os.path.join(merger.train_dir, "annotations.json"))
    merger.save_dataset(val_data, os.path.join(merger.val_dir, "annotations.json"))
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(merged_data['images'])}")
    print(f"Total annotations: {len(merged_data['annotations'])}")
    print(f"Train images: {len(train_data['images'])}")
    print(f"Train annotations: {len(train_data['annotations'])}")
    print(f"Validation images: {len(val_data['images'])}")
    print(f"Validation annotations: {len(val_data['annotations'])}")
    
    # Count actual images in directories
    train_images = len(os.listdir(os.path.join(merger.train_dir, "images")))
    val_images = len(os.listdir(os.path.join(merger.val_dir, "images")))
    
    print(f"\nActual images copied:")
    print(f"Train directory: {train_images} images")
    print(f"Validation directory: {val_images} images")
    
    print("\nDataset structure created at:", output_dir)

if __name__ == "__main__":
    main()
