# South African ID Book Detection Training with Detectron2 (Colab Version)

## 1. Mount Google Drive and Install Dependencies

# Mount Google Drive
from google.colab import drive
import os

# Check if drive is already mounted
if os.path.exists('/content/drive'):
    print("Drive is already mounted. Unmounting first...")
    !fusermount -u /content/drive
    !rm -rf /content/drive

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Verify the dataset directory exists
print("\nVerifying paths...")
DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"
DATASET_ROOT = os.path.join(DRIVE_ROOT, "dj_dataset")

if os.path.exists(DATASET_ROOT):
    print(f"✓ Found dataset directory: {DATASET_ROOT}")
    print("Contents:")
    for item in os.listdir(DATASET_ROOT):
        print(f"  - {item}")
else:
    print(f"✗ Dataset directory not found: {DATASET_ROOT}")

# Install required packages
import subprocess
import sys

def run_pip_install(package):
    print(f"\nInstalling {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    print(f"Successfully installed {package}")

# Install required packages
print("Starting dependency installation...")
packages = [
    'torch',
    'torchvision',
    'scikit-learn',  # Added for train/val split
    'git+https://github.com/facebookresearch/detectron2.git'
]

for package in packages:
    run_pip_install(package)
print("\nAll dependencies installed successfully!")

## 2. Import Libraries and Setup Environment

```python
print("Importing required libraries...")

import os
import cv2
import json
import random
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict
import shutil
from sklearn.model_selection import train_test_split

from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.engine import hooks

# Setup logger
setup_logger()

# Print environment information
print("\nEnvironment Information:")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"OpenCV version: {cv2.__version__}")
print("\nAll libraries imported successfully!")
```

## 3. Setup Project Directory and Prepare Dataset

```python
# Set project directory paths for Colab
DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"
DATASET_ROOT = os.path.join(DRIVE_ROOT, "dj_dataset")
LABEL_STUDIO_EXPORT = os.path.join(DATASET_ROOT, "result.json")  # result.json is directly in dj_dataset
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")  # images folder is directly in dj_dataset

# Output directories
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "val")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "model_output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

def prepare_dataset_structure():
    """Create necessary directories for the dataset."""
    directories = [
        TRAIN_DIR, VAL_DIR,
        os.path.join(TRAIN_DIR, "images"),
        os.path.join(VAL_DIR, "images"),
        OUTPUT_DIR, LOG_DIR,
        IMAGES_DIR  # Add images directory to be created
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def verify_and_fix_image_dimensions(coco_data, images_dir):
    """Verify and fix image dimensions in COCO annotations."""
    print("\nVerifying and fixing image dimensions...")
    fixed_count = 0
    
    for img in coco_data['images']:
        img_path = os.path.join(images_dir, img['file_name'])
        if os.path.exists(img_path):
            # Read actual image dimensions
            image = cv2.imread(img_path)
            if image is not None:
                actual_height, actual_width = image.shape[:2]
                
                # Check if dimensions need to be updated
                if img['width'] != actual_width or img['height'] != actual_height:
                    print(f"Fixing dimensions for {img['file_name']}")
                    print(f"  Annotation: {img['width']}x{img['height']}")
                    print(f"  Actual: {actual_width}x{actual_height}")
                    img['width'] = actual_width
                    img['height'] = actual_height
                    fixed_count += 1
            else:
                print(f"Warning: Could not read image {img_path}")
        else: # The else statement was incorrectly indented
            print(f"Warning: Image not found {img_path}")
    
    print(f"Fixed dimensions for {fixed_count} images")
    return coco_data

def process_label_studio_export():
    """Process Label Studio export and split into train/val sets."""
    print("\nProcessing Label Studio export...")
    
    # Verify Label Studio export exists
    if not os.path.exists(LABEL_STUDIO_EXPORT):
        print(f"\nChecking directory contents:")
        print(f"DATASET_ROOT: {DATASET_ROOT}")
        if os.path.exists(DATASET_ROOT):
            print("Files in dataset directory:")
            for item in os.listdir(DATASET_ROOT):
                print(f"  - {item}")
        else:
            print("Dataset directory does not exist!")
        raise FileNotFoundError(f"Label Studio export not found at: {LABEL_STUDIO_EXPORT}")
    
    print(f"Found Label Studio export at: {LABEL_STUDIO_EXPORT}")
    
    # Read Label Studio export
    with open(LABEL_STUDIO_EXPORT, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Successfully loaded COCO format export file")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    
    # Fix category IDs to start from 1
    category_id_map = {}
    for idx, category in enumerate(coco_data['categories'], start=1):
        category_id_map[category['id']] = idx
        category['id'] = idx
    
    # Update category IDs in annotations
    for ann in coco_data['annotations']:
        ann['category_id'] = category_id_map[ann['category_id']]
    
    # Fix image dimensions and clean up paths
    for img in coco_data['images']:
        # Clean up file paths - remove 'images\' or 'images/' prefix and normalize path
        img['file_name'] = os.path.basename(img['file_name'].replace('\\', '/'))
        
        # Get full image path
        img_path = os.path.join(IMAGES_DIR, img['file_name'])
        if os.path.exists(img_path):
            # Read actual image dimensions
            image = cv2.imread(img_path)
            if image is not None:
                actual_height, actual_width = image.shape[:2]
                if img['width'] != actual_width or img['height'] != actual_height:
                    print(f"Fixing dimensions for {img['file_name']}")
                    print(f"  Annotation: {img['width']}x{img['height']}")
                    print(f"  Actual: {actual_width}x{actual_height}")
                    img['width'] = actual_width
                    img['height'] = actual_height
    
    # Get image filenames
    image_files = [img['file_name'] for img in coco_data['images']]
    print(f"\nFound {len(image_files)} images in annotations")
    
    # Print first few filenames for verification
    print("\nFirst few image filenames after cleanup:")
    for filename in image_files[:5]:
        print(f"- {filename}")
    
    # Verify all images exist
    missing_images = []
    for img_file in image_files:
        if not os.path.exists(os.path.join(IMAGES_DIR, img_file)):
            missing_images.append(img_file)
    
    if missing_images:
        print("\nWarning: Following images are missing:")
        for img in missing_images[:5]:
            print(f"- {img}")
        if len(missing_images) > 5:
            print(f"... and {len(missing_images) - 5} more")
    
    # Split into train/val sets
    image_ids = [img['id'] for img in coco_data['images']]
    train_ids, val_ids = train_test_split(
        image_ids, test_size=0.2, random_state=42
    )
    
    print(f"\nSplit: {len(train_ids)} training images, {len(val_ids)} validation images")
    
    # Create train/val splits
    splits = {
        'train': (train_ids, TRAIN_DIR),
        'val': (val_ids, VAL_DIR)
    }
    
    for split_name, (split_ids, split_dir) in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        # Create COCO format annotations for this split
        split_data = {
            "images": [img for img in coco_data['images'] if img['id'] in split_ids],
            "categories": coco_data['categories'],
            "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in split_ids]
        }
        
        # Save annotations
        ann_path = os.path.join(split_dir, "annotations.json")
        with open(ann_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"✓ Saved annotations to {ann_path}")
        
        # Copy images
        split_filenames = [img['file_name'] for img in split_data['images']]
        copied_count = 0
        for img_file in split_filenames:
            src = os.path.join(IMAGES_DIR, img_file)
            dst = os.path.join(split_dir, "images", img_file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_count += 1
            else:
                print(f"Warning: Could not find image {img_file}")
        
        print(f"✓ Copied {copied_count}/{len(split_filenames)} images for {split_name}")
        print(f"✓ Split contains {len(split_data['images'])} images and {len(split_data['annotations'])} annotations")

# Prepare dataset
print("\nPreparing dataset structure...")
prepare_dataset_structure()
process_label_studio_export()

# Verify the dataset structure
def verify_dataset():
    """Verify the prepared dataset structure and contents."""
    print("\nVerifying dataset structure:")
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        
        # Check annotation file
        ann_path = os.path.join(DATASET_ROOT, split, "annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)
            print(f"✓ Annotations: {len(data['images'])} images, {len(data['annotations'])} annotations")
            
            # Print some statistics about annotations
            categories = {}
            for ann in data['annotations']:
                cat_id = ann['category_id']
                categories[cat_id] = categories.get(cat_id, 0) + 1
            
            print("\nAnnotations per category:")
            for cat in data['categories']:
                count = categories.get(cat['id'], 0)
                print(f"- {cat['name']}: {count}")
        else:
            print(f"✗ Missing annotations file: {ann_path}")
        
        # Check images directory
        img_dir = os.path.join(DATASET_ROOT, split, "images")
        if os.path.exists(img_dir):
            # Check for both jpg and jpeg files
            jpg_images = list(Path(img_dir).glob("*.jpg"))
            jpeg_images = list(Path(img_dir).glob("*.jpeg"))
            total_images = len(jpg_images) + len(jpeg_images)
            print(f"\n✓ Images directory: {total_images} images")
            print(f"  - {len(jpg_images)} .jpg files")
            print(f"  - {len(jpeg_images)} .jpeg files")
        else:
            print(f"✗ Missing images directory: {img_dir}")

# Run verification
verify_dataset()

def inspect_annotations():
    """Inspect the structure of annotations to check for keypoints."""
    print("\nInspecting annotation structure...")
    
    with open(LABEL_STUDIO_EXPORT, 'r') as f:
        data = json.load(f)
    
    # Check categories structure
    print("\nCategory structure:")
    if len(data['categories']) > 0:
        print(json.dumps(data['categories'][0], indent=2))
    
    # Check annotation structure
    print("\nAnnotation structure:")
    if len(data['annotations']) > 0:
        print(json.dumps(data['annotations'][0], indent=2))
        
        # Check if annotations contain keypoints
        has_keypoints = any('keypoints' in ann for ann in data['annotations'])
        print(f"\nDo annotations contain keypoints? {'Yes' if has_keypoints else 'No'}")

# Add this after loading the data
print("\nInspecting annotation structure...")
inspect_annotations()
```

## 4. Register and Verify Dataset

```python
import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import shutil

def rotate_point(x, y, cx, cy, angle_degrees):
    """Rotate a point around a center point."""
    angle_radians = math.radians(angle_degrees)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    
    # Translate point to origin
    dx = x - cx
    dy = y - cy
    
    # Rotate point
    rotated_x = dx * cos_angle - dy * sin_angle
    rotated_y = dx * sin_angle + dy * cos_angle
    
    # Translate back
    return cx + rotated_x, cy + rotated_y

def get_rotated_box_corners(x, y, width, height, rotation_degrees):
    """Get the four corners of a rotated rectangle."""
    # Center point of the box
    cx = x + width/2
    cy = y + height/2
    
    # Get the four corners (relative to center)
    corners = [
        (x, y),  # top-left
        (x + width, y),  # top-right
        (x + width, y + height),  # bottom-right
        (x, y + height)  # bottom-left
    ]
    
    # Rotate each corner
    rotated_corners = [
        rotate_point(corner[0], corner[1], cx, cy, rotation_degrees)
        for corner in corners
    ]
    
    return rotated_corners

def convert_to_coco_keypoints(input_path, output_dir, images_dir):
    """Convert Label Studio annotations to COCO format with keypoints."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read Label Studio annotations
    with open(input_path, 'r') as f:
        label_studio_data = json.load(f)
    
    # Initialize COCO format structure
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define categories (assuming these are your label classes)
    categories = {
        "surname": 1,
        "names": 2,
        "sex": 3,
        "nationality": 4,
        "identity_number": 5,
        "date_of_birth": 6,
        "country_of_birth": 7,
        "status": 8,
        "photo": 9,
        "id_number": 10,
        "surname_header": 11,
        "names_header": 12,
        "sex_header": 13,
        "nationality_header": 14,
        "id_number_header": 15
    }
    
    # Add categories to COCO format
    for name, id in categories.items():
        coco_format["categories"].append({
            "id": id,
            "name": name,
            "supercategory": "id_field",
            "keypoints": ["top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner"],
            "skeleton": [[1,2], [2,3], [3,4], [4,1]]  # Connecting the corners
        })
    
    # Process each image
    annotation_id = 1
    processed_images = set()
    
    for item in label_studio_data:
        image_filename = os.path.basename(item["image"])
        if image_filename in processed_images:
            continue
            
        processed_images.add(image_filename)
        
        # Add image info
        image_id = len(coco_format["images"]) + 1
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": item["bbox"][0]["original_width"],
            "height": item["bbox"][0]["original_height"]
        })
        
        # Process each bbox annotation
        for bbox in item["bbox"]:
            # Get category
            if "rectanglelabels" not in bbox or not bbox["rectanglelabels"]:
                continue
            category_name = bbox["rectanglelabels"][0]
            if category_name not in categories:
                continue
                
            # Calculate rotated corners
            corners = get_rotated_box_corners(
                bbox["x"] * bbox["original_width"] / 100,  # Convert percentage to pixels
                bbox["y"] * bbox["original_height"] / 100,
                bbox["width"] * bbox["original_width"] / 100,
                bbox["height"] * bbox["original_height"] / 100,
                bbox["rotation"]
            )
            
            # Flatten corners and add visibility (2 = visible)
            keypoints = []
            for corner in corners:
                keypoints.extend([corner[0], corner[1], 2])
            
            # Create COCO annotation
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": categories[category_name],
                "bbox": [
                    bbox["x"] * bbox["original_width"] / 100,
                    bbox["y"] * bbox["original_height"] / 100,
                    bbox["width"] * bbox["original_width"] / 100,
                    bbox["height"] * bbox["original_height"] / 100
                ],
                "area": (bbox["width"] * bbox["original_width"] / 100) * 
                       (bbox["height"] * bbox["original_height"] / 100),
                "segmentation": [],
                "iscrowd": 0,
                "keypoints": keypoints,
                "num_keypoints": 4
            })
            annotation_id += 1
    
    # Split into train and validation sets
    image_ids = [img["id"] for img in coco_format["images"]]
    train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)
    
    # Create train and val datasets
    for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
        split_data = {
            "images": [img for img in coco_format["images"] if img["id"] in split_ids],
            "annotations": [ann for ann in coco_format["annotations"] if ann["image_id"] in split_ids],
            "categories": coco_format["categories"]
        }
        
        # Save split
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Create images directory for split
        split_images_dir = os.path.join(output_dir, split_name, "images")
        os.makedirs(split_images_dir, exist_ok=True)
        
        # Copy images
        for img in split_data["images"]:
            src_path = os.path.join(images_dir, img["file_name"])
            dst_path = os.path.join(split_images_dir, img["file_name"])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

def fix_annotation_paths(json_file):
    """Fix file paths in annotation file to use correct format."""
    print(f"\nFixing paths in {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get the directory containing the images
    img_dir = os.path.dirname(json_file).replace('annotations.json', 'images')
    
    # Fix paths in annotations
    for image in data['images']:
        # Get just the filename without any path
        filename = os.path.basename(image['file_name'].replace('\\', '/'))
        # Set the correct path
        image['file_name'] = os.path.join(img_dir, filename)
    
    # Save the fixed annotations
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Fixed {len(data['images'])} image paths")
    return data

def register_datasets():
    """Register the datasets with Detectron2."""
    print("\nRegistering datasets with Detectron2...")
    
    for split in ['train', 'val']:
        name = f"sa_id_{split}"
        json_file = os.path.join(DATASET_ROOT, split, "annotations.json")
        image_root = os.path.join(DATASET_ROOT, split, "images")
        
        # Verify files exist before registration
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Annotations file not found: {json_file}")
        if not os.path.exists(image_root):
            raise FileNotFoundError(f"Images directory not found: {image_root}")
        
        # Clear any existing registration
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
        if name in MetadataCatalog:
            MetadataCatalog.remove(name)
        
        # Load categories from annotations
        with open(json_file, 'r') as f:
            data = json.load(f)
            categories = {cat['id']: cat['name'] for cat in data['categories']}
            thing_classes = [categories[i] for i in sorted(categories.keys())]
        
        # Register dataset
        register_coco_instances(
            name,
            {"thing_classes": thing_classes},
            json_file,
            image_root
        )
        
        print(f"✓ Registered {name} dataset with {len(thing_classes)} categories")
        print("Categories:", thing_classes)

def visualize_samples(dataset_name, num_samples=3):
    """Visualize a few samples from the dataset."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    for d in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        plt.figure(figsize=(15, 10))
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.axis('off')
        plt.title(f"Sample from {dataset_name}")
        plt.show()

# Main execution
print("\nPreparing dataset structure...")
prepare_dataset_structure()
process_label_studio_export()

# Verify the dataset structure
verify_dataset()

# Register datasets after preparation
print("\nRegistering datasets...")
register_datasets()

# Visualize some samples
print("\nVisualizing training samples:")
visualize_samples("sa_id_train")
```

## 5. Training Configuration and Model Training

```python
class CocoTrainer(DefaultTrainer):
    """Custom trainer to evaluate on validation set during training."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        # Add evaluation hook
        hooks_list.append(
            hooks.EvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                lambda: self.test_with_TTA(self.cfg, self.model),
                self.cfg.DATASETS.TEST[0]
            )
        )
        return hooks_list
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        """Run inference with test-time augmentation."""
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir):
    """Setup Detectron2 configuration."""
    cfg = get_cfg()
    
    # Use Faster R-CNN configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Dataset config
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    
    # Solver config
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size (adjust based on your GPU memory)
    cfg.SOLVER.BASE_LR = 0.00025  # Learning rate
    cfg.SOLVER.MAX_ITER = 5000    # Maximum iterations
    cfg.SOLVER.STEPS = []         # Learning rate decay steps
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iterations
    
    # Model config
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Number of classes (15 in your case)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # Testing threshold
    
    # Input config
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # Images will be resized to this min size for training
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Evaluation config
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate every 1000 iterations
    
    # Output config
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

# Configure and train model
print("\nConfiguring model...")
cfg = setup_cfg(
    train_dataset_name="sa_id_train",
    val_dataset_name="sa_id_val",
    num_classes=15,  # Your dataset has 15 categories
    output_dir=OUTPUT_DIR
)

print("\nModel Configuration:")
print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"Training iterations: {cfg.SOLVER.MAX_ITER}")
print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Start training
print("\nStarting training...")
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

## 6. Save and Export Model

```python
# Save final model
print("\nSaving final model...")
final_model_path = os.path.join(OUTPUT_DIR, "model_final.pth")
print(f"Saving model to: {final_model_path}")

# Create a backup copy in Google Drive
drive_backup_path = os.path.join(DRIVE_ROOT, "model_backup", "model_final.pth")
os.makedirs(os.path.dirname(drive_backup_path), exist_ok=True)
shutil.copy2(final_model_path, drive_backup_path)
print(f"Created backup in Google Drive: {drive_backup_path}")

# Export model config
cfg_path = os.path.join(OUTPUT_DIR, "model_config.yaml")
with open(cfg_path, "w") as f:
    f.write(cfg.dump())
print(f"Saved model configuration to: {cfg_path}")
```

## 7. Run Inference

from detectron2.utils.visualizer import ColorMode
import cv2

def run_inference(image_path, confidence_threshold=0.5):
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Detection confidence threshold (0-1)
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Create predictor
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    outputs = predictor(img)
    
    # Visualize results
    v = Visualizer(img[:, :, ::-1],
                  metadata=MetadataCatalog.get("sa_id_val"),
                  instance_mode=ColorMode.IMAGE_BW)
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.title(f"Predictions (confidence >= {confidence_threshold})")
    plt.show()

    # Print predictions
    instances = outputs["instances"].to("cpu")
    print("\nDetections:")
    for i in range(len(instances)):
        score = instances.scores[i].item()
        label = instances.pred_classes[i].item()
        class_name = MetadataCatalog.get("sa_id_val").thing_classes[label]
        box = instances.pred_boxes[i].tensor[0].tolist()
        print(f"- {class_name}: {score:.2%} confidence")
        print(f"  Box: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")

# Example usage - run on a validation image
print("Running inference on sample validation image...")
val_images_dir = os.path.join(VAL_DIR, "images")
sample_image = os.path.join(val_images_dir, os.listdir(val_images_dir)[0])
run_inference(sample_image, confidence_threshold=0.5)

# Function to run inference on multiple images
def batch_inference(image_dir, confidence_threshold=0.5, max_images=5):
    """
    Run inference on multiple images in a directory.
    
    Args:
        image_dir: Directory containing images
        confidence_threshold: Detection confidence threshold (0-1)
        max_images: Maximum number of images to process
    """
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files[:max_images]:
        print(f"\nProcessing: {image_file}")
        image_path = os.path.join(image_dir, image_file)
        run_inference(image_path, confidence_threshold)

print("\nRunning batch inference on validation set...")
batch_inference(val_images_dir, confidence_threshold=0.5, max_images=3)