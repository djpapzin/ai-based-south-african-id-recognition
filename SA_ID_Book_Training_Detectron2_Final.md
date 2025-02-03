# South African ID Book Detection Training with Detectron2

## 1. Install Dependencies

```python
import subprocess
import sys

def run_pip_install(package):
    print(f"\nInstalling {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    print(f"Successfully installed {package}")

# Install packages
print("Starting dependency installation...")
run_pip_install('torch')
run_pip_install('torchvision')
run_pip_install('git+https://github.com/facebookresearch/detectron2.git')
print("\nAll dependencies installed successfully!")
```

## 2. Import Libraries and Setup Environment

```python
print("Importing required libraries...")

import os
import cv2
import json
import random
import numpy as np
import torch
import shutil
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer

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

## 3. Mount Google Drive and Setup Project Directory

```python
from google.colab import drive

def mount_google_drive():
    """Mount Google Drive and ensure connection."""
    # Check if already mounted
    if os.path.exists('/content/drive') and os.path.ismount('/content/drive'):
        print("✓ Google Drive is already mounted")
        return True
    
    try:
        # Mount drive
        print("Mounting Google Drive... (Accept the authorization when prompted)")
        drive.mount('/content/drive', force_remount=True)
        
        # Verify mounting
        if os.path.exists('/content/drive') and os.path.ismount('/content/drive'):
            print("✓ Google Drive mounted successfully!")
            return True
        else:
            print("✗ Failed to verify Google Drive mounting")
            return False
    except Exception as e:
        print(f"✗ Error mounting Google Drive: {str(e)}")
        return False

# Mount drive and set project directory
if mount_google_drive():
    PROJECT_DIR = "/content/drive/MyDrive/Kwantu/Machine Learning"
    if os.path.exists(PROJECT_DIR):
        print(f"✓ Project directory found: {PROJECT_DIR}")
    else:
        print(f"✗ Project directory not found: {PROJECT_DIR}")
        print("Please make sure your project is in the correct location in Google Drive")
else:
    raise RuntimeError("Failed to mount Google Drive. Please check your connection and try again.")

# Check GPU availability
use_gpu = torch.cuda.is_available()
print(f"\nUsing {'GPU' if use_gpu else 'CPU'} for training")
if use_gpu:
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## 4. Dataset Setup and Validation

```python
# Set up paths
print("\nSetting up dataset paths...")
DATASET_ROOT = os.path.join(PROJECT_DIR, "merged_dataset")
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_output")

# Print paths for verification
print(f"\nVerifying paths:")
print(f"Dataset root: {DATASET_ROOT}")
print(f"Training path: {TRAIN_PATH}")
print(f"Validation path: {VAL_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

# Create required directories
for path in [TRAIN_PATH, VAL_PATH, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
    print(f"✓ Verified/created directory: {path}")

# Define paths
TRAIN_JSON = os.path.join(TRAIN_PATH, "annotations.json")
VAL_JSON = os.path.join(VAL_PATH, "annotations.json")
TRAIN_IMGS = os.path.join(TRAIN_PATH, "images")
VAL_IMGS = os.path.join(VAL_PATH, "images")

# Verify files
print("\nVerifying dataset files...")
missing_files = []
for path in [TRAIN_JSON, VAL_JSON, TRAIN_IMGS, VAL_IMGS]:
    if os.path.exists(path):
        print(f"✓ Found: {path}")
    else:
        print(f"✗ Missing: {path}")
        missing_files.append(path)

if missing_files:
    print("\nError: Missing required files:")
    for file in missing_files:
        print(f"- {file}")
    raise FileNotFoundError(f"Required paths not found: {', '.join(missing_files)}")

# Load and validate class names from annotations
def get_class_names_from_annotations(json_path):
    """Extract class names from COCO annotations file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Sort categories by id to ensure consistent order
        categories = sorted(data['categories'], key=lambda x: x['id'])
        return [cat['name'] for cat in categories]
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        return None

print("\nValidating class names from annotations...")
train_classes = get_class_names_from_annotations(TRAIN_JSON)
val_classes = get_class_names_from_annotations(VAL_JSON)

if train_classes is None or val_classes is None:
    raise RuntimeError("Failed to read class names from annotations")

if train_classes != val_classes:
    print("\nWarning: Class names differ between training and validation sets!")
    print("\nTraining classes:", train_classes)
    print("Validation classes:", val_classes)
    raise ValueError("Training and validation class names must match")

# Use the actual class names from the dataset
CLASS_NAMES = train_classes
print("\nDetected classes:", CLASS_NAMES)
```

## 5. Model Configuration

```python
def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir, use_gpu=True):
    """Setup Detectron2 configuration."""
    print("\nSetting up Detectron2 configuration...")
    cfg = get_cfg()
    
    # Load base configuration
    base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    print(f"Loading base configuration: {base_config}")
    cfg.merge_from_file(model_zoo.get_config_file(base_config))
    
    # Dataset configuration
    print("\nConfiguring dataset settings...")
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    print(f"Training dataset: {train_dataset_name}")
    print(f"Validation dataset: {val_dataset_name}")
    print(f"Number of dataloader workers: {cfg.DATALOADER.NUM_WORKERS}")
    
    # Model configuration
    print("\nConfiguring model architecture...")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    print(f"Pre-trained weights: {os.path.basename(cfg.MODEL.WEIGHTS)}")
    print(f"Number of classes: {num_classes}")
    print(f"Detection threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    
    # Training configuration
    print("\nConfiguring training parameters...")
    cfg.SOLVER.IMS_PER_BATCH = 2 if use_gpu else 1
    if not use_gpu:
        cfg.MODEL.DEVICE = "cpu"
        print("Warning: Using CPU for training (this will be slow)")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    print(f"Base learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Maximum iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Learning rate steps: {cfg.SOLVER.STEPS}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Checkpoint period: {cfg.SOLVER.CHECKPOINT_PERIOD}")
    
    # Testing configuration
    print("\nConfiguring evaluation settings...")
    cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
    print(f"Evaluation period: {cfg.TEST.EVAL_PERIOD}")
    print(f"RoI batch size per image: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
    
    # Input settings
    print("\nConfiguring input settings...")
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    print(f"Training size range: {cfg.INPUT.MIN_SIZE_TRAIN} to {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"Testing size range: {cfg.INPUT.MIN_SIZE_TEST} to {cfg.INPUT.MAX_SIZE_TEST}")
    
    # Output configuration
    print("\nConfiguring output settings...")
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    
    # Save configuration
    config_path = os.path.join(cfg.OUTPUT_DIR, "model_config.yaml")
    with open(config_path, "w") as f:
        f.write(cfg.dump())
    print(f"Configuration saved to: {config_path}")
    
    return cfg

# Create output directory if not exists
OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Setup the configuration
cfg = setup_cfg(
    train_dataset_name="sa_id_train",
    val_dataset_name="sa_id_val",
    num_classes=len(CLASS_NAMES),
    output_dir=OUTPUT_DIR,
    use_gpu=torch.cuda.is_available()
)

print("\n✅ Configuration setup complete!")
```

## 6. Dataset Registration and Visualization

```python
# Import required libraries if not already imported
import os
import random
import traceback
import cv2
import json
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

# Use PROJECT_DIR from Section 3
if 'PROJECT_DIR' not in locals():
    raise RuntimeError("Please run Section 3 first to mount Google Drive and set PROJECT_DIR")

print(f"\nUsing project directory: {PROJECT_DIR}")
DATASET_ROOT = os.path.join(PROJECT_DIR, "merged_dataset")
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")

# Define annotation and image paths
TRAIN_JSON = os.path.join(TRAIN_PATH, "annotations.json")
VAL_JSON = os.path.join(VAL_PATH, "annotations.json")
TRAIN_IMGS = os.path.join(TRAIN_PATH, "images")
VAL_IMGS = os.path.join(VAL_PATH, "images")

# Print paths for verification
print("\nVerifying dataset paths:")
print(f"Dataset root: {DATASET_ROOT}")
print(f"Training annotations: {TRAIN_JSON}")
print(f"Validation annotations: {VAL_JSON}")
print(f"Training images: {TRAIN_IMGS}")
print(f"Validation images: {VAL_IMGS}")

def get_class_names_from_annotations(json_path):
    """Extract class names from COCO annotations file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Sort categories by id to ensure consistent order
        categories = sorted(data['categories'], key=lambda x: x['id'])
        return [cat['name'] for cat in categories]
    except Exception as e:
        print(f"Error reading annotations file: {e}")
        return None

def register_and_visualize_datasets(train_json, val_json, train_imgs, val_imgs):
    """
    Register datasets and visualize samples with proper error handling.
    Returns the dataset dictionaries and metadata for further use.
    """
    print("\nRegistering datasets...")
    
    # Verify paths exist
    missing_paths = []
    for path in [train_json, val_json, train_imgs, val_imgs]:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print("\nWarning: The following paths do not exist:")
        for path in missing_paths:
            print(f"- {path}")
        print("\nPlease ensure you have:")
        print("1. Run Section 3 to mount Google Drive")
        print("2. Created the dataset structure in your Google Drive")
        print(f"\nExpected structure in Google Drive:")
        print(f"{PROJECT_DIR}/")
        print("└── merged_dataset/")
        print("    ├── train/")
        print("    │   ├── images/")
        print("    │   └── annotations.json")
        print("    └── val/")
        print("        ├── images/")
        print("        └── annotations.json")
        raise FileNotFoundError("Required dataset paths not found")
    
    # Get class names from annotations
    print("\nExtracting class names from annotations...")
    train_classes = get_class_names_from_annotations(train_json)
    val_classes = get_class_names_from_annotations(val_json)
    
    if train_classes is None or val_classes is None:
        raise RuntimeError("Failed to read class names from annotations")
    
    if train_classes != val_classes:
        print("\nWarning: Class names differ between training and validation sets!")
        print("\nTraining classes:", train_classes)
        print("Validation classes:", val_classes)
        raise ValueError("Training and validation class names must match")
    
    class_names = train_classes
    print("\nDetected classes:", class_names)
    
    # Unregister existing datasets if present
    for d in ["sa_id_train", "sa_id_val"]:
        if d in DatasetCatalog:
            DatasetCatalog.remove(d)
        if d in MetadataCatalog:
            MetadataCatalog.remove(d)
    
    try:
        # Register datasets with empty metadata first
        register_coco_instances("sa_id_train", {}, train_json, train_imgs)
        register_coco_instances("sa_id_val", {}, val_json, val_imgs)
        
        # Set metadata after registration
        metadata = MetadataCatalog.get("sa_id_train")
        metadata.thing_classes = class_names
        MetadataCatalog.get("sa_id_val").thing_classes = class_names
        
        # Load and verify datasets
        train_dicts = DatasetCatalog.get("sa_id_train")
        val_dicts = DatasetCatalog.get("sa_id_val")
        
        print("✓ Datasets registered successfully!")
        print(f"\nDataset Statistics:")
        print(f"- Training images: {len(train_dicts)}")
        print(f"- Validation images: {len(val_dicts)}")
        
        # Count instances per category
        train_instances = {"total": 0}
        for d in train_dicts:
            for ann in d["annotations"]:
                cat_id = ann["category_id"]
                cat_name = class_names[cat_id]
                train_instances[cat_name] = train_instances.get(cat_name, 0) + 1
                train_instances["total"] += 1
        
        print("\nTraining Set Statistics:")
        print(f"Total instances: {train_instances['total']}")
        print("\nInstances per category:")
        for cat_name in class_names:
            if cat_name in train_instances:
                print(f"- {cat_name}: {train_instances[cat_name]}")
        
        # Visualize samples (in Colab)
        print("\nVisualizing random training samples with annotations...")
        
        for d in random.sample(train_dicts, min(3, len(train_dicts))):
            img = cv2.imread(d["file_name"])
            if img is None:
                print(f"Warning: Could not read image {d['file_name']}")
                continue
            
            print(f"\nDisplaying annotations for: {os.path.basename(d['file_name'])}")
            print(f"Number of annotations: {len(d['annotations'])}")
            
            # Create visualizer with custom settings
            visualizer = Visualizer(img[:, :, ::-1], 
                                 metadata=metadata,
                                 scale=1.0,  # Increased scale for better visibility
            )
            
            # Draw instance annotations
            vis = visualizer.draw_dataset_dict(d)
            drawn_img = vis.get_image()[:, :, ::-1]
            
            # Display image with annotations
            plt.figure(figsize=(20, 10))  # Larger figure size
            plt.imshow(drawn_img)
            plt.axis('off')
            
            # Add title with annotation info
            class_counts = {}
            for ann in d["annotations"]:
                class_name = metadata.thing_classes[ann["category_id"]]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            title = "Annotations:\n" + "\n".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            plt.title(title, loc='left', fontsize=10, pad=10)
            
            plt.show()
            plt.close()
        
        return train_dicts, val_dicts, metadata
        
    except Exception as e:
        print(f"\nError during dataset setup: {str(e)}")
        traceback.print_exc()
        raise

# Register and visualize the datasets
train_dicts, val_dicts, metadata = register_and_visualize_datasets(
    train_json=TRAIN_JSON,
    val_json=VAL_JSON,
    train_imgs=TRAIN_IMGS,
    val_imgs=VAL_IMGS
)

# Store class names for later use
CLASS_NAMES = metadata.thing_classes
```

## 6.5 Pre-Training Verification

```python
def verify_training_setup():
    """Verify all requirements are met before starting training."""
    print("\nVerifying training setup...")
    
    # 1. Check Google Drive mounting
    if not os.path.exists('/content/drive') or not os.path.ismount('/content/drive'):
        raise RuntimeError("Google Drive is not mounted. Please run Section 3 first.")
    
    # 2. Check project directory
    if not os.path.exists(PROJECT_DIR):
        raise RuntimeError(f"Project directory not found: {PROJECT_DIR}")
    
    # 3. Verify dataset structure
    required_paths = {
        'Dataset Root': DATASET_ROOT,
        'Training Path': TRAIN_PATH,
        'Validation Path': VAL_PATH,
        'Training Images': TRAIN_IMGS,
        'Validation Images': VAL_IMGS,
        'Training Annotations': TRAIN_JSON,
        'Validation Annotations': VAL_JSON
    }
    
    missing_paths = []
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("\n❌ Missing required paths:")
        for path in missing_paths:
            print(f"- {path}")
        print("\nPlease ensure your Google Drive has the following structure:")
        print(f"/content/drive/MyDrive/Kwantu/Machine Learning/")
        print("├── merged_dataset/")
        print("│   ├── train/")
        print("│   │   ├── images/")
        print("│   │   └── annotations.json")
        print("│   └── val/")
        print("│       ├── images/")
        print("│       └── annotations.json")
        raise FileNotFoundError("Required dataset structure is incomplete")
    
    # 4. Check annotations format
    try:
        with open(TRAIN_JSON, 'r') as f:
            train_anns = json.load(f)
        with open(VAL_JSON, 'r') as f:
            val_anns = json.load(f)
        
        required_keys = ['images', 'annotations', 'categories']
        for dataset, anns in [('Training', train_anns), ('Validation', val_anns)]:
            missing_keys = [k for k in required_keys if k not in anns]
            if missing_keys:
                raise KeyError(f"{dataset} annotations missing required keys: {missing_keys}")
    except json.JSONDecodeError:
        raise ValueError("Annotations files are not valid JSON")
    
    # 5. Check image files
    print("\nChecking image files...")
    missing_images = []
    for dataset, anns in [('Training', train_anns), ('Validation', val_anns)]:
        img_dir = TRAIN_IMGS if dataset == 'Training' else VAL_IMGS
        for img in anns['images']:
            img_path = os.path.join(img_dir, os.path.basename(img['file_name']))
            if not os.path.exists(img_path):
                missing_images.append(f"{dataset}: {img['file_name']}")
    
    if missing_images:
        print("\n⚠️ Warning: Some referenced images are missing:")
        for img in missing_images[:5]:
            print(f"- {img}")
        if len(missing_images) > 5:
            print(f"... and {len(missing_images)-5} more")
    
    # 6. Verify GPU availability
    print("\nChecking GPU availability...")
    if not torch.cuda.is_available():
        print("⚠️ Warning: No GPU detected. Training will be slow on CPU.")
    else:
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    
    # 7. Verify dataset registration
    print("\nVerifying dataset registration...")
    if "sa_id_train" not in DatasetCatalog:
        raise RuntimeError("Training dataset not registered. Please run Section 6 first.")
    if "sa_id_val" not in DatasetCatalog:
        raise RuntimeError("Validation dataset not registered. Please run Section 6 first.")
    
    print("\n✅ All verification checks passed! Ready for training.")
    return True

# Run verification before training
verify_training_setup()
```

## 7. Training and Evaluation

```python
# Setup configuration
cfg = setup_cfg(
    train_dataset_name="sa_id_train",
    val_dataset_name="sa_id_val",
    num_classes=len(CLASS_NAMES),
    output_dir=OUTPUT_DIR,
    use_gpu=use_gpu
)

# Create logs directory for TensorBoard
os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

# Start TensorBoard
%load_ext tensorboard
%tensorboard --logdir=os.path.join(OUTPUT_DIR, "logs")

# Train model
print("\nStarting model training...")
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save model
print("\nSaving trained model...")
final_model_path = os.path.join(OUTPUT_DIR, "sa_id_detector_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)

# Save configuration
config_path = os.path.join(OUTPUT_DIR, "model_config.yaml")
with open(config_path, "w") as f:
    f.write(cfg.dump())

# Evaluate model
print("\nEvaluating model...")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("sa_id_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg, trainer.model, evaluators=[evaluator])

# Save summary
summary_path = os.path.join(OUTPUT_DIR, "model_summary.txt")
with open(summary_path, "w") as f:
    f.write("South African ID Detection Model Summary\n")
    f.write("=====================================\n\n")
    f.write(f"Training completed at: {datetime.datetime.now()}\n")
    f.write(f"Model saved at: {final_model_path}\n")
    f.write(f"Configuration saved at: {config_path}\n\n")
    f.write("Dataset Statistics:\n")
    f.write(f"- Training images: {len(DatasetCatalog.get('sa_id_train'))}\n")
    f.write(f"- Validation images: {len(DatasetCatalog.get('sa_id_val'))}\n")
    f.write("\nModel Configuration:\n")
    f.write(f"- Base model: Faster R-CNN R50-FPN\n")
    f.write(f"- Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}\n")
    f.write(f"- Detection threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}\n")

print(f"\nTraining completed! Summary saved to: {summary_path}")
```

## 8. Inference Function

```python
def run_inference(image_path, predictor, confidence_threshold=0.7):
    """Run inference on a single image and visualize results."""
    print(f"\nRunning inference on: {image_path}")
    
    # Read image
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    # Run inference
    outputs = predictor(im)
    
    # Visualize results
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("sa_id_val"),
                   scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Print results
    print("\nInference Results:")
    print(f"Number of detections: {len(outputs['instances'])}")
    if len(outputs['instances']) > 0:
        scores = outputs['instances'].scores.tolist()
        classes = [
            MetadataCatalog.get("sa_id_val").thing_classes[i] 
            for i in outputs['instances'].pred_classes.tolist()
        ]
        for cls, score in zip(classes, scores):
            if score >= confidence_threshold:
                print(f"- {cls}: {score:.3f}")
    
    return outputs, out.get_image()[:, :, ::-1]