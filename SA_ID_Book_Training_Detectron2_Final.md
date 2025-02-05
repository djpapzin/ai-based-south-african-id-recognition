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
        else: 
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

## 4. Register and Verify Dataset

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import shutil
import random

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

# Comprehensive Data Validation
print('\n=== Data Integrity Checks ===')

# 1. Verify category consistency
max_category = max([k['category_id'] for d in DatasetCatalog.get("sa_id_train") for k in d['annotations']])
print(f'Max category ID: {max_category} (should be {len(MetadataCatalog.get("sa_id_train").thing_classes)-1})')

# 2. Check image file existence
for d in DatasetCatalog.get("sa_id_train"):
    if not os.path.exists(d['file_name']):
        print(f'Missing image file: {d["file_name"]}')

# 3. Validate bounding box formats
for idx, d in enumerate(DatasetCatalog.get("sa_id_train")):
    for anno in d['annotations']:
        bbox = anno['bbox']
        if len(bbox) != 4:
            print(f'Invalid bbox format in image {d["image_id"]}: {bbox}')
        if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
            print(f'Invalid bbox dimensions in image {d["image_id"]}: {bbox}')

# 4. Verify segmentation formats
for d in DatasetCatalog.get("sa_id_train"):
    for anno in d['annotations']:
        if 'segmentation' in anno:
            for seg in anno['segmentation']:
                if len(seg) < 6 or len(seg) % 2 != 0:
                    print(f'Invalid segmentation in image {d["image_id"]}')

## 5. Training Configuration and Model Training
class CocoTrainer(DefaultTrainer):
    """Custom trainer to evaluate on validation set during training."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks_list = super().build_hooks()
        # # Add evaluation hook
        # hooks_list.append(
        #     hooks.EvalHook(
        #         self.cfg.TEST.EVAL_PERIOD,
        #         lambda: self.test_with_TTA(self.cfg, self.model),
        #         self.cfg.DATASETS.TEST[0]
        #     )
        # )
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

    # Use COCO pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # Dataset config
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)

    # Solver parameters for quick demo
    cfg.SOLVER.IMS_PER_BATCH = 8  # Try increasing batch size (monitor GPU memory)
    cfg.SOLVER.BASE_LR = 0.001    # Slightly higher learning rate (use scheduler later)
    cfg.SOLVER.MAX_ITER = 500     # Reduced iterations for quick training
    cfg.SOLVER.CHECKPOINT_PERIOD = 200  # Save checkpoints more frequently (reduce frequency of checkpoints)
    cfg.SOLVER.WARMUP_ITERS = 50  # Reduced warmup period

    # Evaluation settings
    cfg.TEST.EVAL_PERIOD = 500    # Evaluate less frequently (e.g., 500 or 1000) - AVOID FREQUENT EVALUATIONS

    # Model config
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    metadata = MetadataCatalog.get(train_dataset_name)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)  # Should be 12
    print(f'\n\n=== Model Config ===')
    print(f'Number of classes configured: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}')
    print(f'Dataset categories: {len(metadata.thing_classes)}')
    assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == len(metadata.thing_classes), "Class count mismatch!"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Initialize new layers properly
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
    cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7

    # Input config - keep COCO standard sizes
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704) # Reduce size range if possible - EXPERIMENT!
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Enable data augmentation
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 2 # Increase data loading workers (AFTER data loading is optimized)

    # Evaluation config
    cfg.TEST.EVAL_PERIOD = 500  # Reduce evaluation frequency

    # Output config
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

# Configure and train model
print("\nConfiguring model...")

# Set CUDA options for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Get number of categories from the dataset
train_metadata = MetadataCatalog.get("sa_id_train")
num_classes = len(train_metadata.thing_classes)

# Remove any existing model checkpoints
if os.path.exists(OUTPUT_DIR):
    print("Cleaning up old checkpoints...")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.pth'):
            os.remove(os.path.join(OUTPUT_DIR, f))

cfg = setup_cfg(
    train_dataset_name="sa_id_train",
    val_dataset_name="sa_id_val",
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
        print("1. Restart the runtime")
        print("2. Run all cells from the beginning")
        print("3. If the error persists, try reducing the batch size to 1")
    raise
    
## 6. Save and Export Model

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
metadata = MetadataCatalog.get("sa_id_train")  # Get metadata from training dataset
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

## 7. Run Inference

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install torch torchvision
!pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Required imports
import os
import cv2
import torch
import random
import json
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import model_zoo
from google.colab.patches import cv2_imshow

# Register dataset metadata
thing_classes = [
    'bottom_left_corner', 'bottom_right_corner', 'citizenship_status',
    'country_of_birth', 'date_of_birth', 'face', 'id_document', 'id_number',
    'names', 'nationality', 'sex', 'signature', 'surname', 'top_left_corner',
    'top_right_corner'
]

# Register metadata if not already registered
if "sa_id_val" not in MetadataCatalog:
    MetadataCatalog.get("sa_id_val").set(thing_classes=thing_classes)

def run_inference(image_path, confidence_threshold=0.5, cfg=None):
    """Run inference on a single image"""
    if cfg is None:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Kwantu/Machine Learning/model_output/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        predictor = DefaultPredictor(cfg)
        # Fix weights_only warning
        predictor.model.load_state_dict(
            torch.load(cfg.MODEL.WEIGHTS, 
                      map_location=cfg.MODEL.DEVICE,
                      weights_only=True)
        )

    print(f"\nRunning inference on {cfg.MODEL.DEVICE.upper()}")
    predictor = DefaultPredictor(cfg)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Run inference with updated autocast
    if cfg.MODEL.DEVICE == 'cuda':
        with torch.amp.autocast('cuda'):
            outputs = predictor(image)
    else:
        outputs = predictor(image)
    
    # Visualize results
    v = Visualizer(image[:, :, ::-1],
                  metadata=MetadataCatalog.get("sa_id_val"),
                  scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display result
    cv2_imshow(out.get_image()[:, :, ::-1])
    
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
    
    if cfg.MODEL.DEVICE == 'cuda':
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    return outputs

def save_labeled_segments(image_path, outputs, save_dir):
    """Save detected segments with their labels"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Create directory for this image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.join(save_dir, base_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # Get instances and metadata
    instances = outputs["instances"].to("cpu")
    metadata = MetadataCatalog.get("sa_id_val")
    field_counts = {}
    
    # Process detections
    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor[0].numpy().astype(int)
        label = instances.pred_classes[i].item()
        score = instances.scores[i].item()
        class_name = metadata.thing_classes[label]
        
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
        cv2.imwrite(os.path.join(image_dir, filename), segment)
    
    # Save metadata
    metadata_dict = {
        "timestamp": datetime.now().isoformat(),
        "detections": {
            f"{class_name}{'_'+str(field_counts[class_name]) if field_counts[class_name]>1 else ''}.jpg": {
                "class": class_name,
                "confidence": float(score),
                "bbox": [int(x) for x in instances.pred_boxes[i].tensor[0].tolist()]
            }
            for i, class_name in enumerate([metadata.thing_classes[label] for label in instances.pred_classes])
        }
    }
    
    with open(os.path.join(image_dir, "detection_metadata.json"), 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"Saved {len(instances)} segments from {os.path.basename(image_path)}")
    print(f"Output directory: {image_dir}")

def batch_inference(image_dir, confidence_threshold=0.5, max_images=None, save_dir=None):
    """Run inference on multiple images"""
    # Get and sample image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # If max_images is specified, randomly sample images
    if max_images is not None:
        selected_files = random.sample(image_files, min(max_images, len(image_files)))
    else:
        selected_files = image_files
    
    print(f"Found {len(image_files)} images, processing {len(selected_files)}")
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving labeled segments to: {save_dir}")
    
    # Setup configuration once
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Kwantu/Machine Learning/model_output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    
    for image_file in selected_files:
        print(f"\nProcessing: {image_file}")
        image_path = os.path.join(image_dir, image_file)
        outputs = run_inference(image_path, confidence_threshold, cfg)
        
        if save_dir:
            save_labeled_segments(image_path, outputs, save_dir)

# Example usage - will run when cell is executed
if __name__ == "__main__":
    # Full paths to directories
    EXAMPLE_IDS_DIR = "/content/drive/MyDrive/Kwantu/Machine Learning/dj_dataset/example_ids"
    VAL_DIR = "/content/drive/MyDrive/Kwantu/Machine Learning/dj_dataset/val"
    SAVE_DIR = "/content/drive/MyDrive/Kwantu/Machine Learning/dj_dataset/detected_segments"
    
    # Process example IDs
    print("\nRunning batch inference on example IDs...")
    if os.path.exists(EXAMPLE_IDS_DIR):
        batch_inference(EXAMPLE_IDS_DIR, 
                       confidence_threshold=0.5,
                       max_images=None,  # Process all images
                       save_dir=os.path.join(SAVE_DIR, "examples"))
    else:
        print(f"Warning: Example IDs directory not found at {EXAMPLE_IDS_DIR}")
    
    # Process validation images
    print("\nRunning batch inference on validation images...")
    if os.path.exists(VAL_DIR):
        batch_inference(VAL_DIR,
                       confidence_threshold=0.5,
                       max_images=None,  # Process all images
                       save_dir=os.path.join(SAVE_DIR, "validation"))
    else:
        print(f"Warning: Validation directory not found at {VAL_DIR}")

## 8. OCR Processing with Tesseract

import pytesseract
import json
from PIL import Image
import os
from datetime import datetime

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_segments_with_ocr(segments_dir, output_json):
    """
    Process all segmented images with OCR and save results to JSON.
    
    Args:
        segments_dir (str): Directory containing segmented images
        output_json (str): Path to save JSON output
    """
    results = {}
    
    # Process each image in the directory
    for filename in os.listdir(segments_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Extract metadata from filename
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        original_image = parts[0]
        field_type = parts[1]
        confidence = float(parts[2])
        
        # Read image with PIL
        image_path = os.path.join(segments_dir, filename)
        try:
    # Read image
            image = Image.open(image_path)
            
            # Configure OCR settings based on field type
            config = ''
            if field_type == 'id_number':
                config = '--psm 7 -c tessedit_char_whitelist=0123456789'
            elif field_type in ['date_of_birth', 'date_of_issue']:
                config = '--psm 7 -c tessedit_char_whitelist=0123456789/'
            else:
                config = '--psm 7'
            
            # Perform OCR
            text = pytesseract.image_to_string(image, config=config).strip()
            
            # Store results
            if original_image not in results:
                results[original_image] = {
                    'timestamp': datetime.now().isoformat(),
                    'fields': {}
                }
            
            results[original_image]['fields'][field_type] = {
                'text': text,
                'detection_confidence': confidence,
                'segment_file': filename
            }
            
            print(f"Processed {filename}: {text}")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Save results to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_json}")
    return results

# Process validation segments
print("\nProcessing validation segments with OCR...")
val_segments_dir = os.path.join("dj_dataset", "detected_segments", "validation")
val_output_json = os.path.join("dj_dataset", "ocr_results", "validation_results.json")
val_results = process_segments_with_ocr(val_segments_dir, val_output_json)

# Display sample results
print("\nSample OCR Results:")
for image_id, data in list(val_results.items())[:2]:  # Show first 2 images
    print(f"\nImage: {image_id}")
    for field, info in data['fields'].items():
        print(f"- {field}: {info['text']} (confidence: {info['detection_confidence']:.2f})")

# Standalone Inference Cell - Complete Setup
import os
import cv2
import torch
import random
import json
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import model_zoo
from google.colab.patches import cv2_imshow

# Register dataset metadata
thing_classes = [
    'bottom_left_corner', 'bottom_right_corner', 'citizenship_status',
    'country_of_birth', 'date_of_birth', 'face', 'id_document', 'id_number',
    'names', 'nationality', 'sex', 'signature', 'surname', 'top_left_corner',
    'top_right_corner'
]

# Register metadata if not already registered
if "sa_id_val" not in MetadataCatalog:
    MetadataCatalog.get("sa_id_val").set(thing_classes=thing_classes)

def run_inference(image_path, confidence_threshold=0.5, cfg=None):
    """Run inference on a single image"""
    if cfg is None:
        # Setup configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Kwantu/Machine Learning/model_output/model_final.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        predictor = DefaultPredictor(cfg)
        # Fix weights_only warning
        predictor.model.load_state_dict(
            torch.load(cfg.MODEL.WEIGHTS, 
                      map_location=cfg.MODEL.DEVICE,
                      weights_only=True)
        )

    print(f"\nRunning inference on {cfg.MODEL.DEVICE.upper()}")
    predictor = DefaultPredictor(cfg)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Run inference with updated autocast
    if cfg.MODEL.DEVICE == 'cuda':
        with torch.amp.autocast('cuda'):
            outputs = predictor(image)
    else:
        outputs = predictor(image)
    
    # Visualize results
    v = Visualizer(image[:, :, ::-1],
                  metadata=MetadataCatalog.get("sa_id_val"),
                  scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Display result
    cv2_imshow(out.get_image()[:, :, ::-1])
    
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
    
    if cfg.MODEL.DEVICE == 'cuda':
        print(f"\nGPU Memory Used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    return outputs

def save_labeled_segments(image_path, outputs, save_dir):
    """Save detected segments with their labels"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Create directory for this image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.join(save_dir, base_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # Get instances and metadata
    instances = outputs["instances"].to("cpu")
    metadata = MetadataCatalog.get("sa_id_val")
    field_counts = {}
    
    # Process detections
    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor[0].numpy().astype(int)
        label = instances.pred_classes[i].item()
        score = instances.scores[i].item()
        class_name = metadata.thing_classes[label]
        
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
        cv2.imwrite(os.path.join(image_dir, filename), segment)
    
    # Save metadata
    metadata_dict = {
        "timestamp": datetime.now().isoformat(),
        "detections": {
            f"{class_name}{'_'+str(field_counts[class_name]) if field_counts[class_name]>1 else ''}.jpg": {
                "class": class_name,
                "confidence": float(score),
                "bbox": [int(x) for x in instances.pred_boxes[i].tensor[0].tolist()]
            }
            for i, class_name in enumerate([metadata.thing_classes[label] for label in instances.pred_classes])
        }
    }
    
    with open(os.path.join(image_dir, "detection_metadata.json"), 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"Saved {len(instances)} segments from {os.path.basename(image_path)}")
    print(f"Output directory: {image_dir}")

def batch_inference(image_dir, confidence_threshold=0.5, max_images=None, save_dir=None):
    """Run inference on multiple images"""
    # Get and sample image files
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # If max_images is specified, randomly sample images
    if max_images is not None:
        selected_files = random.sample(image_files, min(max_images, len(image_files)))
    else:
        selected_files = image_files
    
    print(f"Found {len(image_files)} images, processing {len(selected_files)}")
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving labeled segments to: {save_dir}")
    
    # Setup configuration once
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/Kwantu/Machine Learning/model_output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    
    for image_file in selected_files:
        print(f"\nProcessing: {image_file}")
        image_path = os.path.join(image_dir, image_file)
        outputs = run_inference(image_path, confidence_threshold, cfg)
        
        if save_dir:
            save_labeled_segments(image_path, outputs, save_dir)

# Example usage:
if __name__ == "__main__":
    # Paths
    DRIVE_ROOT = "/content/drive/MyDrive/Kwantu/Machine Learning"
    DATASET_ROOT = os.path.join(DRIVE_ROOT, "dj_dataset")
    VAL_DIR = os.path.join(DATASET_ROOT, "val")
    SAVE_DIR = os.path.join(DATASET_ROOT, "detected_segments")
    
    # Run inference on validation images
    print("\nRunning batch inference on validation images...")
    batch_inference(VAL_DIR, 
                   confidence_threshold=0.5,
                   max_images=None,  # Process all images
                   save_dir=os.path.join(SAVE_DIR, "validation"))