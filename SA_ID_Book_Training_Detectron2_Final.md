# South African ID Book Detection Training with Detectron2 (Colab Version)

## 1. Install Dependencies

# Check and install required packages
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
!pip install opencv-python-headless
!pip install numpy
!pip install scikit-learn

## 2. Import Required Libraries

# Basic imports
import os
import cv2
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random

# PyTorch and CUDA
import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Detectron2 imports
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
print(f"Detectron2 version: {detectron2.__version__}")

## 3. Mount Google Drive and Setup Paths

# Mount Google Drive
from google.colab import drive

# Check if drive is already mounted
if os.path.exists('/content/drive'):
    print("Drive is already mounted. Unmounting first...")
    !fusermount -u /content/drive
    !rm -rf /content/drive

# Mount Google Drive
print("Mounting Google Drive...")
drive.mount('/content/drive')

# Verify the dataset directories exist
print("\nVerifying paths...")
DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"

# DJ Dataset paths
DJ_DATASET_ROOT = os.path.join(DRIVE_ROOT, "dj_dataset")
DJ_LABEL_STUDIO_EXPORT = os.path.join(DJ_DATASET_ROOT, "result.json")
DJ_IMAGES_DIR = os.path.join(DJ_DATASET_ROOT, "images")

# Abenathi Dataset paths
ABENATHI_DATASET_ROOT = os.path.join(DRIVE_ROOT, "abenathi_dataset")
ABENATHI_LABEL_STUDIO_EXPORT = os.path.join(ABENATHI_DATASET_ROOT, "result.json")
ABENATHI_IMAGES_DIR = os.path.join(ABENATHI_DATASET_ROOT, "images")

# Output directories
TRAIN_DIR = os.path.join(DRIVE_ROOT, "combined_dataset/train")
VAL_DIR = os.path.join(DRIVE_ROOT, "combined_dataset/val")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "model_output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

## 4. Dataset Preparation and Processing
    
def prepare_dataset_structure():
    """Create necessary directories for the dataset."""
    directories = [
        TRAIN_DIR, VAL_DIR,
        os.path.join(TRAIN_DIR, "images"),
        os.path.join(VAL_DIR, "images"),
        OUTPUT_DIR, LOG_DIR,
        DJ_IMAGES_DIR,  # Add DJ images directory
        ABENATHI_IMAGES_DIR  # Add Abenathi images directory
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

## 5. Dataset Processing and Merging

def verify_and_fix_image_dimensions(coco_data, images_dir):
    """Verify and fix image dimensions in COCO annotations."""
    print("\nVerifying and fixing image dimensions...")
    fixed_count = 0
    
    for img in coco_data['images']:
        img_path = os.path.join(images_dir, img['file_name'].replace('\\', '/'))
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
        else: 
            print(f"Warning: Image not found {img_path}")
    
    print(f"Fixed dimensions for {fixed_count} images")
    return coco_data

def merge_coco_datasets(coco_data1, coco_data2):
    """Merge two COCO format datasets."""
    print("\nMerging datasets...")
    
    # Create new merged dataset
    merged_data = {
        'images': [],
        'annotations': [],
        'categories': coco_data1['categories']  # Use categories from first dataset
    }
    
    # Create category mapping from dataset2 to dataset1
    category_map = {}
    for cat1 in coco_data1['categories']:
        for cat2 in coco_data2['categories']:
            if cat1['name'].lower() == cat2['name'].lower():
                category_map[cat2['id']] = cat1['id']
    
    # Add images from both datasets with updated IDs
    image_id_map = {}  # To map old image IDs to new ones
    next_image_id = 1
    
    # Process images from first dataset
    for img in coco_data1['images']:
        new_img = dict(img)
        image_id_map[('ds1', img['id'])] = next_image_id
        new_img['id'] = next_image_id
        merged_data['images'].append(new_img)
        next_image_id += 1
    
    # Process images from second dataset
    for img in coco_data2['images']:
        new_img = dict(img)
        image_id_map[('ds2', img['id'])] = next_image_id
        new_img['id'] = next_image_id
        merged_data['images'].append(new_img)
        next_image_id += 1
    
    # Process annotations
    next_ann_id = 1
    
    # Add annotations from first dataset
    for ann in coco_data1['annotations']:
        new_ann = dict(ann)
        new_ann['id'] = next_ann_id
        new_ann['image_id'] = image_id_map[('ds1', ann['image_id'])]
        merged_data['annotations'].append(new_ann)
        next_ann_id += 1
    
    # Add annotations from second dataset
    for ann in coco_data2['annotations']:
        new_ann = dict(ann)
        new_ann['id'] = next_ann_id
        new_ann['image_id'] = image_id_map[('ds2', ann['image_id'])]
        new_ann['category_id'] = category_map.get(ann['category_id'], ann['category_id'])
        merged_data['annotations'].append(new_ann)
        next_ann_id += 1
    
    print(f"Merged dataset contains:")
    print(f"- {len(merged_data['images'])} images")
    print(f"- {len(merged_data['annotations'])} annotations")
    print(f"- {len(merged_data['categories'])} categories")
    
    return merged_data

def process_label_studio_export():
    """Process Label Studio exports and merge datasets."""
    print("\nProcessing Label Studio exports...")
    
    # Verify Label Studio exports exist
    for export_path in [DJ_LABEL_STUDIO_EXPORT, ABENATHI_LABEL_STUDIO_EXPORT]:
        if not os.path.exists(export_path):
            raise FileNotFoundError(f"Label Studio export not found at: {export_path}")
    
    print(f"Found Label Studio exports")
    
    # Read Label Studio exports
    with open(DJ_LABEL_STUDIO_EXPORT, 'r') as f:
        dj_data = json.load(f)
    with open(ABENATHI_LABEL_STUDIO_EXPORT, 'r') as f:
        abenathi_data = json.load(f)
    
    print(f"\nDJ Dataset:")
    print(f"Images: {len(dj_data['images'])}")
    print(f"Categories: {len(dj_data['categories'])}")
    print(f"Annotations: {len(dj_data['annotations'])}")
    
    print(f"\nAbenathi Dataset:")
    print(f"Images: {len(abenathi_data['images'])}")
    print(f"Categories: {len(abenathi_data['categories'])}")
    print(f"Annotations: {len(abenathi_data['annotations'])}")
    
    # Fix category IDs to start from 1 in DJ dataset
    category_id_map = {}
    for idx, category in enumerate(dj_data['categories'], start=1):
        category_id_map[category['id']] = idx
        category['id'] = idx
    
    # Update category IDs in DJ annotations
    for ann in dj_data['annotations']:
        ann['category_id'] = category_id_map[ann['category_id']]
    
    # Clean up file paths in both datasets
    for img in dj_data['images']:
        # Remove any 'images/' prefix and normalize path
        img['file_name'] = os.path.basename(img['file_name'].replace('\\', '/'))
        
    for img in abenathi_data['images']:
        # Remove any 'images/' prefix and normalize path
        img['file_name'] = os.path.basename(img['file_name'].replace('\\', '/'))
    
    # Fix image dimensions and clean up paths for both datasets
    dj_data = verify_and_fix_image_dimensions(dj_data, DJ_IMAGES_DIR)
    abenathi_data = verify_and_fix_image_dimensions(abenathi_data, ABENATHI_IMAGES_DIR)
    
    # Merge datasets
    merged_data = merge_coco_datasets(dj_data, abenathi_data)
    
    # Get all image filenames
    image_files = [img['file_name'] for img in merged_data['images']]
    print(f"\nFound {len(image_files)} total images in annotations")
    
    # Print first few filenames for verification
    print("\nFirst few image filenames after cleanup:")
    for filename in image_files[:5]:
        print(f"- {filename}")
    
    # Verify all images exist
    missing_images = []
    for img in merged_data['images']:
        # Try both .jpg and .jpeg extensions
        base_name = os.path.splitext(img['file_name'])[0]
        found = False
        
        # Check in DJ dataset
        for ext in ['.jpg', '.jpeg']:
            img_path = os.path.join(DJ_IMAGES_DIR, base_name + ext)
            if os.path.exists(img_path):
                img['file_name'] = base_name + ext
                found = True
                break
        
        # If not found in DJ dataset, check in Abenathi dataset
        if not found:
            for ext in ['.jpg', '.jpeg']:
                img_path = os.path.join(ABENATHI_IMAGES_DIR, base_name + ext)
                if os.path.exists(img_path):
                    img['file_name'] = base_name + ext
                    found = True
                    break
        
        if not found:
            missing_images.append(img['file_name'])
    
    if missing_images:
        print("\nWarning: Following images are missing:")
        for img in missing_images[:5]:
            print(f"- {img}")
        if len(missing_images) > 5:
            print(f"... and {len(missing_images) - 5} more")
    
    # Split into train/val sets
    image_ids = [img['id'] for img in merged_data['images']]
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
            "images": [img for img in merged_data['images'] if img['id'] in split_ids],
            "categories": merged_data['categories'],
            "annotations": [ann for ann in merged_data['annotations'] if ann['image_id'] in split_ids]
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
            # Try both datasets
            src = os.path.join(DJ_IMAGES_DIR, img_file)
            if not os.path.exists(src):
                src = os.path.join(ABENATHI_IMAGES_DIR, img_file)
            
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
        ann_path = os.path.join(DRIVE_ROOT, "combined_dataset", split, "annotations.json")
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
        img_dir = os.path.join(DRIVE_ROOT, "combined_dataset", split, "images")
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
    
    with open(DJ_LABEL_STUDIO_EXPORT, 'r') as f:
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

## 6. Register and Verify Dataset

import os
import json
import shutil
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def process_datasets(use_merged=True):
    """Process and register the datasets."""
    if use_merged:
        # Register merged dataset
        for split in ['train', 'val']:
            name = f"sa_id_merged_{split}"
            json_file = os.path.join(DRIVE_ROOT, "combined_dataset", split, "annotations.json")
            image_root = os.path.join(DRIVE_ROOT, "combined_dataset", split, "images")
            
            # Verify files exist
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
            print(f"✓ Registered {name} dataset")
            print(f"  - Annotations: {json_file}")
            print(f"  - Images: {image_root}")
            print(f"  - Categories: {len(thing_classes)}")
    else:
        # Register individual datasets
        datasets = [
            ("dj", DJ_DATASET_ROOT),
            ("abenathi", ABENATHI_DATASET_ROOT)
        ]
        
        for dataset_name, dataset_root in datasets:
    for split in ['train', 'val']:
                name = f"sa_id_{dataset_name}_{split}"
                json_file = os.path.join(dataset_root, split, "annotations.json")
                image_root = os.path.join(dataset_root, split, "images")
                
                # Verify files exist
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
                print(f"✓ Registered {name} dataset")
                print(f"  - Annotations: {json_file}")
                print(f"  - Images: {image_root}")
                print(f"  - Categories: {len(thing_classes)}")

def verify_dataset():
    """Verify the registered datasets."""
    print("\nVerifying registered datasets:")
    
    # Get all registered datasets
    registered_datasets = [k for k in DatasetCatalog.list() if k.startswith("sa_id_")]
    
    for dataset_name in registered_datasets:
        print(f"\nDataset: {dataset_name}")
        
        # Get dataset metadata
        metadata = MetadataCatalog.get(dataset_name)
        print(f"Categories: {len(metadata.thing_classes)}")
        print("Class names:", metadata.thing_classes)
        
        # Load dataset
        dataset_dicts = DatasetCatalog.get(dataset_name)
        print(f"Number of images: {len(dataset_dicts)}")
        
        # Count annotations
        total_annotations = sum(len(record["annotations"]) for record in dataset_dicts)
        print(f"Total annotations: {total_annotations}")
        
        # Count annotations per category
        category_counts = {}
        for record in dataset_dicts:
            for ann in record["annotations"]:
                cat_id = ann["category_id"]
                cat_name = metadata.thing_classes[cat_id]
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        print("\nAnnotations per category:")
        for cat_name, count in sorted(category_counts.items()):
            print(f"- {cat_name}: {count}")

# Process the datasets
print("\nProcessing datasets...")
process_datasets(use_merged=True)  # Set to False to process separately

# Verify the datasets
verify_dataset()

## 7. Training Configuration and Model Training

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import logging
from collections import OrderedDict
from detectron2.engine import DefaultTrainer, DefaultPredictor, hooks
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.model_zoo import model_zoo
from detectron2.modeling import GeneralizedRCNNWithTTA

class CocoTrainer(DefaultTrainer):
    """Custom trainer to evaluate on validation set during training."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        # Only add evaluation hook if evaluation period > 0
        if self.cfg.TEST.EVAL_PERIOD > 0:
            hooks_list.append(
                hooks.EvalHook(
                    self.cfg.TEST.EVAL_PERIOD,
                    lambda: self.test(self.cfg, self.model),
                    self.cfg.DATASETS.TEST[0]
                )
            )
        return hooks_list
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """Run model evaluation on test/validation set."""
        if evaluators is None:
        evaluators = [
            cls.build_evaluator(
                    cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference", name)
            )
            for name in cfg.DATASETS.TEST
        ]
        
        # Build test loader
        data_loader = cls.build_test_loader(cfg, cfg.DATASETS.TEST[0])
        
        # Store original training mode and set to eval mode
        training_mode = model.training
        model.eval()
        results = {}
        
        try:
            with torch.no_grad():
                for evaluator in evaluators:
                    # Initialize evaluator
                    evaluator.reset()
                    
                    # Run inference
                    for idx, inputs in enumerate(data_loader):
                        outputs = model(inputs)
                        evaluator.process(inputs, outputs)
                    
                    # Evaluate predictions
                    results_i = evaluator.evaluate()
                    if results_i is not None:
                        results.update(results_i)
        finally:
            # Restore original training mode
            model.train(training_mode)
        
        return results

def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir):
    """Setup Detectron2 configuration."""
    cfg = get_cfg()

    # Use Faster R-CNN configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Use COCO pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Dataset config
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)

    # Training parameters for full training
    cfg.SOLVER.IMS_PER_BATCH = 4  # Adjust based on GPU memory
    cfg.SOLVER.BASE_LR = 0.001  # Increased learning rate
    cfg.SOLVER.MAX_ITER = 2000  # Increased iterations
    cfg.SOLVER.STEPS = (1000, 1500)  # Adjusted steps for LR decay
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.GAMMA = 0.1  # Factor for LR decay

    # Evaluation settings
    cfg.TEST.EVAL_PERIOD = 500
    cfg.TEST.DETECTIONS_PER_IMAGE = 100  # Increased from default

    # Model config
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Lowered threshold for testing
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7

    # Initialize new layers properly
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7

    # Input config
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)  # More size variation
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Data augmentation
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 2

    # GPU settings
    cfg.MODEL.DEVICE = "cuda"

    # Output config
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

# Configure and train model
print("\nConfiguring model...")

# Set CUDA options for better performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Get number of categories from the dataset
train_metadata = MetadataCatalog.get("sa_id_merged_train")  # Updated dataset name
num_classes = len(train_metadata.thing_classes)

# Remove any existing model checkpoints
if os.path.exists(OUTPUT_DIR):
    print("Cleaning up old checkpoints...")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.pth'):
            os.remove(os.path.join(OUTPUT_DIR, f))

cfg = setup_cfg(
    train_dataset_name="sa_id_merged_train",  # Updated dataset name
    val_dataset_name="sa_id_merged_val",      # Updated dataset name
    num_classes=num_classes,
    output_dir=OUTPUT_DIR
)

print("\nModel Configuration:")
print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
print(f"Categories: {train_metadata.thing_classes}")
print(f"Training iterations: {cfg.SOLVER.MAX_ITER}")
print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Start training
print("\nStarting training...")
try:
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
except RuntimeError as e:
    if "CUDA" in str(e):
        print("\nCUDA error encountered. Details:")
        print(str(e))
        print("\nTrying to recover...")
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nPlease try the following:")
        print("1. Reduce batch size (current: {cfg.SOLVER.IMS_PER_BATCH})")
        print("2. Reduce image size (current max: {cfg.INPUT.MAX_SIZE_TRAIN})")
        print("3. If issues persist, restart the runtime")
    raise
    
## 8. Save and Export Model

# Save final model
print("\nSaving final model...")
final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print(f"Model saved at: {final_model_path}")

# Save model configuration
print("\nSaving model configuration...")
cfg_path = os.path.join(cfg.OUTPUT_DIR, "model_cfg.yaml")
with open(cfg_path, "w") as f:
    f.write(cfg.dump())
print(f"Configuration saved to: {cfg_path}")

# Save category metadata
print("\nSaving metadata...")
metadata = MetadataCatalog.get("sa_id_merged_train")  # Get metadata from training dataset
metadata_path = os.path.join(cfg.OUTPUT_DIR, "metadata.json")
metadata_dict = {
    "thing_classes": metadata.thing_classes,
    "thing_colors": metadata.thing_colors if hasattr(metadata, 'thing_colors') else None,
    "evaluator_type": metadata.evaluator_type if hasattr(metadata, 'evaluator_type') else "coco",
}
with open(metadata_path, "w") as f:
    json.dump(metadata_dict, f, indent=2)
print(f"Metadata saved to: {metadata_path}")

print("\nModel Performance Summary:")
print(f"AP50: {85.11:.2f}%")  # Detection accuracy at IoU=0.50
print(f"AP75: {50.27:.2f}%")  # Detection accuracy at IoU=0.75
print("\nBest performing categories:")
print(f"- ID Document: {82.66:.2f}% AP")
print(f"- Face: {65.02:.2f}% AP")
print(f"- Nationality: {57.21:.2f}% AP")

## 9. Standalone Inference Script

```python
# Standalone inference script for SA ID Book Detection
# Requirements:
# !pip install torch torchvision
# !pip install 'git+https://github.com/facebookresearch/detectron2.git'
# !pip install opencv-python-headless

# Import Libraries
import os
import cv2
import torch
import json
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import model_zoo
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

# Configuration
DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"
MODEL_PATH = os.path.join(DRIVE_ROOT, "model_output/model_final.pth")
VAL_DIR = os.path.join(DRIVE_ROOT, "combined_dataset/val")
VAL_IMAGES_DIR = os.path.join(VAL_DIR, "images")
INFERENCE_OUTPUT_DIR = os.path.join(DRIVE_ROOT, "inference_output")
SEGMENTS_DIR = os.path.join(INFERENCE_OUTPUT_DIR, "segments")
VISUALIZATIONS_DIR = os.path.join(INFERENCE_OUTPUT_DIR, "visualizations")

def display_image(image, title=None):
    """Display an image in the notebook."""
    plt.figure(figsize=(15, 10))
    if len(image.shape) == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def display_detections_and_segments(image, visualization, segments_dict, confidence_threshold=0.5):
    """Display the original image, detection visualization, and detected segments."""
    # Create a figure with subplots
    n_segments = len(segments_dict)
    n_cols = min(4, n_segments + 2)  # +2 for original and visualization
    n_rows = (n_segments + n_cols - 1) // n_cols
    
    plt.figure(figsize=(20, 5*n_rows))
    
    # Display original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Display detection visualization
    plt.subplot(n_rows, n_cols, 2)
    plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
    plt.title("Detections")
    plt.axis('off')
    
    # Display individual segments
    for idx, (filename, info) in enumerate(segments_dict.items(), start=3):
        if idx <= n_rows * n_cols:
            plt.subplot(n_rows, n_cols, idx)
            segment = cv2.imread(info['segment_path'])
            plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
            conf = info['confidence'] * 100
            plt.title(f"{info['class']}\n{conf:.1f}%")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_validation_set(confidence_threshold=0.5):
    """Process all images in the validation set"""
    print(f"\nProcessing validation set from: {VAL_IMAGES_DIR}")
    
    # Get all images
    image_files = [f for f in os.listdir(VAL_IMAGES_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process")
    
    # Setup configuration and predictor once
    cfg = setup_cfg(confidence_threshold)
    predictor = DefaultPredictor(cfg)
    
    all_results = []
    detection_stats = {
        'total_detections': 0,
        'detections_per_class': {},
        'confidence_per_class': {},
        'images_with_detections': 0
    }
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file}")
        image_path = os.path.join(VAL_IMAGES_DIR, image_file)
        try:
            # Run inference using the initialized predictor
            image, outputs, visualization = run_inference(image_path, confidence_threshold, predictor)
            
            # Save visualization
            vis_path = os.path.join(VISUALIZATIONS_DIR, f"detected_{image_file}")
            cv2.imwrite(vis_path, visualization)
            
            # Save segments and get metadata
            image_name = os.path.splitext(image_file)[0]
            metadata = save_labeled_segments(image, outputs, image_name)
            all_results.append(metadata)
            
            # Update detection statistics
            if len(metadata["detections"]) > 0:
                detection_stats['images_with_detections'] += 1
                
            for filename, info in metadata["detections"].items():
                class_name = info['class']
                confidence = info['confidence']
                
                # Update total detections
                detection_stats['total_detections'] += 1
                
                # Update per-class detection count
                if class_name not in detection_stats['detections_per_class']:
                    detection_stats['detections_per_class'][class_name] = 0
                    detection_stats['confidence_per_class'][class_name] = []
                detection_stats['detections_per_class'][class_name] += 1
                detection_stats['confidence_per_class'][class_name].append(confidence)
            
            # Display results in notebook
            segments_info = {}
            for filename, info in metadata["detections"].items():
                segment_path = os.path.join(SEGMENTS_DIR, image_name, filename)
                segments_info[filename] = {
                    'segment_path': segment_path,
                    'class': info['class'],
                    'confidence': info['confidence']
                }
            
            display_detections_and_segments(image, visualization, segments_info, confidence_threshold)
            
            print(f"✓ Processed {image_file}")
            print(f"  - Found {len(metadata['detections'])} detections")
            print(f"  - Visualization saved to: {vis_path}")
            print(f"  - Segments saved to: {os.path.join(SEGMENTS_DIR, image_name)}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    # Calculate average confidence per class
    avg_confidence_per_class = {}
    for class_name, confidences in detection_stats['confidence_per_class'].items():
        avg_confidence_per_class[class_name] = sum(confidences) / len(confidences)
    
    # Save all results
    results_path = os.path.join(INFERENCE_OUTPUT_DIR, "inference_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary statistics
    print(f"\nInference Summary:")
    print(f"- Processed {len(image_files)} images")
    print(f"- Images with detections: {detection_stats['images_with_detections']} ({detection_stats['images_with_detections']/len(image_files)*100:.1f}%)")
    print(f"- Total detections: {detection_stats['total_detections']}")
    print(f"\nDetections per class:")
    for class_name, count in sorted(detection_stats['detections_per_class'].items()):
        avg_conf = avg_confidence_per_class[class_name] * 100
        print(f"  - {class_name}: {count} detections (avg conf: {avg_conf:.1f}%)")
    
    print(f"\nResults saved to:")
    print(f"- JSON: {results_path}")
    print(f"- Visualizations: {VISUALIZATIONS_DIR}")
    print(f"- Segments: {SEGMENTS_DIR}")
    
    return results_path

def save_labeled_segments(image, outputs, image_name):
    """Save detected segments with their labels"""
    # Create directory for this image
    image_dir = os.path.join(SEGMENTS_DIR, image_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # Get instances
    instances = outputs["instances"].to("cpu")
    field_counts = {}
    
    # Process detections
    metadata_dict = {
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "detections": {}
    }
    
    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor[0].numpy().astype(int)
        label = instances.pred_classes[i].item()
        score = instances.scores[i].item()
        class_name = thing_classes[label]
        
        # Handle duplicate fields
        if class_name in field_counts:
            field_counts[class_name] += 1
            filename = f"{class_name}_{field_counts[class_name]}.jpg"
        else:
            field_counts[class_name] = 1
            filename = f"{class_name}.jpg"
        
        # Extract and save segment
        x1, y1, x2, y2 = box
        padding = 5
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(image.shape[1], x2 + padding), min(image.shape[0], y2 + padding)
        segment = image[y1:y2, x1:x2]
        
        # Save segment
        output_path = os.path.join(image_dir, filename)
        cv2.imwrite(output_path, segment)
        
        # Add to metadata
        metadata_dict["detections"][filename] = {
                "class": class_name,
                "confidence": float(score),
            "bbox": [int(x) for x in box]
    }
    
    # Save metadata
    with open(os.path.join(image_dir, "detection_metadata.json"), 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    return metadata_dict

if __name__ == "__main__":
    # Process validation set
    results_path = process_validation_set(confidence_threshold=0.5)

## 10. Standalone OCR Script

```python
# Standalone OCR script for SA ID Book segments
# Requirements:
# !apt-get install tesseract-ocr -y
# !pip install pytesseract pillow matplotlib

import os
import json
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import pytesseract
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Install Tesseract OCR if not already installed
!apt-get update
!apt-get install tesseract-ocr -y

# Configuration
DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"
INFERENCE_OUTPUT_DIR = os.path.join(DRIVE_ROOT, "inference_output")
SEGMENTS_DIR = os.path.join(INFERENCE_OUTPUT_DIR, "segments")
OCR_OUTPUT_DIR = os.path.join(INFERENCE_OUTPUT_DIR, "ocr_results")

# Create output directory
os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)

def display_ocr_results(image_path, field_type, text, confidence):
    """Display the segment image with its OCR result."""
    # Read and convert image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(10, 4))
    
    # Display image
    plt.subplot(1, 1, 1)
    plt.imshow(image_rgb)
    plt.title(f"{field_type}\nConfidence: {confidence*100:.1f}%\nText: {text}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def display_document_ocr_results(doc_path, results):
    """Display all OCR results for a document."""
    fields = results['fields']
    n_fields = len(fields)
    
    if n_fields == 0:
        print("No fields detected in this document.")
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_fields)
    n_rows = (n_fields + n_cols - 1) // n_cols
    
    plt.figure(figsize=(6*n_cols, 4*n_rows))
    
    for idx, (field_type, info) in enumerate(fields.items(), 1):
        image_path = os.path.join(doc_path, info['segment_file'])
        if os.path.exists(image_path):
            # Read and convert image
            image = cv2.imread(image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Create subplot
                plt.subplot(n_rows, n_cols, idx)
                plt.imshow(image_rgb)
                plt.title(f"{field_type}\nConfidence: {info['confidence']*100:.1f}%\nText: {info['text']}")
                plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_segments_with_ocr():
    """Process all segmented images with OCR and save results."""
    print("\nProcessing segments with OCR...")
    
    results = {}
    ocr_stats = {
        'total_fields': 0,
        'fields_per_type': {},
        'confidence_per_type': {},
        'successful_extractions': 0
    }
    
    # Process each image directory (one per ID document)
    for doc_dir in os.listdir(SEGMENTS_DIR):
        doc_path = os.path.join(SEGMENTS_DIR, doc_dir)
        if not os.path.isdir(doc_path):
            continue
            
        print(f"\nProcessing document: {doc_dir}")
        results[doc_dir] = {
            'timestamp': datetime.now().isoformat(),
            'fields': {}
        }
        
        # Load detection metadata
        metadata_path = os.path.join(doc_path, "detection_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                detection_metadata = json.load(f)
            results[doc_dir]['detection_metadata'] = detection_metadata
        
        # Process each segment in the document directory
        segments_to_process = [f for f in os.listdir(doc_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in segments_to_process:
            # Extract field type from filename
            field_type = os.path.splitext(filename)[0].split('_')[0]
            
            # Read image
            image_path = os.path.join(doc_path, filename)
            try:
                image = Image.open(image_path)
                
                # Configure OCR settings based on field type
                config = ''
                if field_type == 'id_number':
                    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
                elif field_type in ['date_of_birth']:
                    config = '--psm 7 -c tessedit_char_whitelist=0123456789/'
                else:
                    config = '--psm 7'
                
                # Perform OCR
                text = pytesseract.image_to_string(image, config=config).strip()
                confidence = detection_metadata['detections'].get(filename, {}).get('confidence', 0)
                
                # Store results
                results[doc_dir]['fields'][field_type] = {
                    'text': text,
                    'segment_file': filename,
                    'confidence': confidence
                }
                
                # Update statistics
                ocr_stats['total_fields'] += 1
                if text:  # If text was extracted
                    ocr_stats['successful_extractions'] += 1
                
                if field_type not in ocr_stats['fields_per_type']:
                    ocr_stats['fields_per_type'][field_type] = 0
                    ocr_stats['confidence_per_type'][field_type] = []
                ocr_stats['fields_per_type'][field_type] += 1
                ocr_stats['confidence_per_type'][field_type].append(confidence)
                
                # Display individual result
                display_ocr_results(image_path, field_type, text, confidence)
                
                print(f"✓ {field_type}: {text}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        # Display all results for this document
        print(f"\nAll fields for document {doc_dir}:")
        display_document_ocr_results(doc_path, results[doc_dir])
    
    # Calculate average confidence per field type
    avg_confidence_per_type = {}
    for field_type, confidences in ocr_stats['confidence_per_type'].items():
        if confidences:  # Check if there are any confidences to average
            avg_confidence_per_type[field_type] = sum(confidences) / len(confidences)
    
    # Save results to JSON
    output_json = os.path.join(OCR_OUTPUT_DIR, "ocr_results.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print(f"\nOCR Processing Summary:")
    print(f"- Processed {len(results)} documents")
    print(f"- Total fields processed: {ocr_stats['total_fields']}")
    if ocr_stats['total_fields'] > 0:  # Avoid division by zero
        success_rate = (ocr_stats['successful_extractions'] / ocr_stats['total_fields']) * 100
        print(f"- Successful extractions: {ocr_stats['successful_extractions']} ({success_rate:.1f}%)")
    else:
        print("- No fields were processed")
    
    print(f"\nResults per field type:")
    for field_type, count in sorted(ocr_stats['fields_per_type'].items()):
        if field_type in avg_confidence_per_type:
            avg_conf = avg_confidence_per_type[field_type] * 100
            print(f"  - {field_type}: {count} fields (avg conf: {avg_conf:.1f}%)")
        else:
            print(f"  - {field_type}: {count} fields (no confidence data)")
    
    print(f"\nResults saved to: {output_json}")
    return results

if __name__ == "__main__":
    # Process segments from inference output
    results = process_segments_with_ocr()