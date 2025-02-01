# South African ID Book Detection Training with Detectron2

This notebook trains a Detectron2 model for detecting South African ID books using a pre-split dataset.
The notebook supports both GPU and CPU environments, with automatic device selection.

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

## 2. Import Libraries

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

## 3. Mount Google Drive

```python
def mount_google_drive():
    """Mount Google Drive and ensure connection."""
from google.colab import drive
    import os
    
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

# Mount drive and verify project directory
if mount_google_drive():
    PROJECT_DIR = "/content/drive/MyDrive/Kwantu/Machine Learning"
    if os.path.exists(PROJECT_DIR):
        print(f"✓ Project directory found: {PROJECT_DIR}")
    else:
        print(f"✗ Project directory not found: {PROJECT_DIR}")
        print("Please make sure your project is in the correct location in Google Drive")
else:
    raise RuntimeError("Failed to mount Google Drive. Please check your connection and try again.")
```

## 4. GPU Check Function

```python
def check_gpu():
    """Check GPU availability and print device information."""
    print("\nChecking GPU availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        return True
    return False

use_gpu = check_gpu()
print(f"\nUsing {'GPU' if use_gpu else 'CPU'} for training")
```

## 5. Training Classes and Configuration

```python
class CocoTrainer(DefaultTrainer):
    """Custom trainer class with COCO evaluator."""
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir, use_gpu=True):
    """Setup Detectron2 configuration."""
    print("\nSetting up configuration...")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # Don't filter out images with no annotations
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True  # Group images with similar aspect ratios
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set testing threshold
    
    # Training configuration
    if use_gpu:
        cfg.SOLVER.IMS_PER_BATCH = 2
    else:
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.DEVICE = "cpu"
    
    # Solver settings
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    
    # Test settings
    cfg.TEST.EVAL_PERIOD = 500
    
    # ROI settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25  # Balance positive/negative samples
    
    # Input settings - Add more augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    cfg.INPUT.BRIGHTNESS = 0.8
    cfg.INPUT.CONTRAST = 0.8
    cfg.INPUT.SATURATION = 0.8
    cfg.INPUT.HUE = 0.1
    
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("\nConfiguration setup completed successfully!")
    print("\nConfiguration summary:")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"ROI batch size per image: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
    print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"Empty annotations filtered: {cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS}")
    print(f"Data augmentation enabled: Random flip, brightness, contrast, saturation, and hue")
    return cfg
```

## 6. Dataset Investigation

```python
print("\nSetting up dataset paths for investigation...")
DATASET_ROOT = os.path.join(PROJECT_DIR, "merged_dataset")
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
TRAIN_JSON = os.path.join(TRAIN_PATH, "annotations_fixed.json")

# Verify paths
if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")
if not os.path.exists(TRAIN_JSON):
    raise FileNotFoundError(f"Training annotations not found: {TRAIN_JSON}")

print(f"✓ Using dataset root: {DATASET_ROOT}")
print(f"✓ Using training annotations: {TRAIN_JSON}")

# Load and examine the annotations file
print("\nInvestigating dataset annotations...")
import json
with open(TRAIN_JSON, 'r') as f:
    annotations = json.load(f)

# Print basic statistics
print(f"\nDataset Statistics:")
print(f"Total images in annotations: {len(annotations['images'])}")
print(f"Total annotations: {len(annotations['annotations'])}")

# Check for images without annotations
image_ids_with_annots = set(ann['image_id'] for ann in annotations['annotations'])
all_image_ids = set(img['id'] for img in annotations['images'])
images_without_annots = all_image_ids - image_ids_with_annots

print(f"\nImages without annotations: {len(images_without_annots)}")
if images_without_annots:
    print("Warning: Found images without annotations!")

# Check annotation validity
invalid_annotations = []
for ann in annotations['annotations']:
    # Check for empty or invalid bounding boxes
    if 'bbox' not in ann or len(ann['bbox']) != 4:
        invalid_annotations.append(('missing_bbox', ann))
    elif any(coord < 0 for coord in ann['bbox']):
        invalid_annotations.append(('negative_coords', ann))
    elif ann['bbox'][2] * ann['bbox'][3] == 0:  # width * height = 0
        invalid_annotations.append(('zero_area', ann))
    elif 'category_id' not in ann:
        invalid_annotations.append(('missing_category', ann))

print(f"\nInvalid annotations found: {len(invalid_annotations)}")
if invalid_annotations:
    print("\nWarning: Found invalid annotations!")
    print("Sample of invalid annotations:")
    for error_type, ann in invalid_annotations[:5]:
        print(f"Error type: {error_type}")
        print(f"Annotation: {ann}\n")

# Check category distribution
category_counts = {}
for ann in annotations['annotations']:
    cat_id = ann['category_id']
    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

print("\nCategory distribution:")
for cat_id, count in category_counts.items():
    print(f"Category {cat_id}: {count} instances")

print("\nDataset investigation completed!")

# Raise warning if issues found
if images_without_annots or invalid_annotations:
    print("\n⚠️ WARNING: Dataset issues detected! Please review the above results before proceeding.")
            else:
    print("\n✅ Dataset looks good! No major issues detected.")
```

## 7. TensorBoard Setup

```python
# Load and start TensorBoard
%load_ext tensorboard
%tensorboard --logdir=model_output/logs
```

## 8. Dataset Registration and Path Setup

```python
# Set up paths using PROJECT_DIR
print("\nSetting up dataset paths...")
DATASET_ROOT = os.path.join(PROJECT_DIR, "merged_dataset")
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_output")

# Ensure directories exist
for path in [TRAIN_PATH, VAL_PATH, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)
    print(f"✓ Verified/created directory: {path}")

# Define annotation and image paths
TRAIN_JSON = os.path.join(TRAIN_PATH, "annotations_fixed.json")
VAL_JSON = os.path.join(VAL_PATH, "annotations_fixed.json")
TRAIN_IMGS = os.path.join(TRAIN_PATH, "images")
VAL_IMGS = os.path.join(VAL_PATH, "images")

# Verify dataset files
print("\nVerifying dataset files...")
for path in [TRAIN_JSON, VAL_JSON, TRAIN_IMGS, VAL_IMGS]:
    if os.path.exists(path):
        print(f"✓ Found: {path}")
        else:
        print(f"✗ Missing: {path}")
        raise FileNotFoundError(f"Required path not found: {path}")

# Register datasets
print("\nRegistering datasets...")
try:
    # Unregister if already registered
    for d in ["sa_id_train", "sa_id_val"]:
        if d in DatasetCatalog:
            DatasetCatalog.remove(d)
        if d in MetadataCatalog:
            MetadataCatalog.remove(d)
    
    # Register datasets
    register_coco_instances(
        "sa_id_train",
        {},
        TRAIN_JSON,
        TRAIN_IMGS
    )
    register_coco_instances(
        "sa_id_val",
        {},
        VAL_JSON,
        VAL_IMGS
    )
    print("✓ Datasets registered successfully!")

    # Set metadata
    CLASS_NAMES = [
        "id_number", "surname", "names", "nationality",
        "country_of_birth", "status", "sex", "date_of_birth",
        "id_number_barcode", "identity_number_back",
        "control_number", "district_of_birth", "issue_date",
        "id_photo", "id_book"
    ]
    
    for d in ["sa_id_train", "sa_id_val"]:
        MetadataCatalog.get(d).set(thing_classes=CLASS_NAMES)
    print("✓ Metadata set successfully!")

    # Verify registration
    train_dicts = DatasetCatalog.get("sa_id_train")
    val_dicts = DatasetCatalog.get("sa_id_val")
    print(f"\nRegistered {len(train_dicts)} training images")
    print(f"Registered {len(val_dicts)} validation images")

except Exception as e:
    print(f"Error during dataset registration: {str(e)}")
    raise

# Setup configuration
print("\nSetting up model configuration...")
cfg = setup_cfg(
    train_dataset_name="sa_id_train",
    val_dataset_name="sa_id_val",
    num_classes=len(CLASS_NAMES),
    output_dir=OUTPUT_DIR,
    use_gpu=use_gpu
)
print("✓ Configuration completed!")
```

## 9. Training

```python
print("\nStarting model training...")
print("\nTraining configuration:")
print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"ROI batch size per image: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
print("\nTraining started. This may take several hours...")
trainer.train()
print("\nTraining completed successfully!")
```

## 10. Model Evaluation

```python
print("\nStarting model evaluation...")

# Load the trained model
print("Loading trained model...")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)
print("Model loaded successfully!")

# Run evaluation
print("\nRunning evaluation on validation dataset...")
evaluator = COCOEvaluator("sa_id_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg, trainer.model, evaluators=[evaluator])
print("Evaluation completed!")

# Visualize some predictions
print("\nVisualizing predictions on sample images...")
val_dataset_dicts = DatasetCatalog.get("sa_id_val")
for d in random.sample(val_dataset_dicts, 3):
    print(f"\nProcessing image: {os.path.basename(d['file_name'])}")
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("sa_id_val"),
                   scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(15, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.show()
    
    # Print prediction details
    print(f"Number of detections: {len(outputs['instances'])}")
    if len(outputs['instances']) > 0:
        print("Confidence scores:", outputs['instances'].scores.tolist())
        print("Predicted classes:", [
            MetadataCatalog.get("sa_id_val").thing_classes[i] 
            for i in outputs['instances'].pred_classes.tolist()
        ])
```

## 11. Save Model

```python
print("\nSaving trained model...")
final_model_path = os.path.join(OUTPUT_DIR, "sa_id_detector_final.pth")
torch.save(trainer.model.state_dict(), final_model_path)
print(f"Model saved successfully to: {final_model_path}")

# Save configuration
config_path = os.path.join(OUTPUT_DIR, "model_config.yaml")
print(f"\nSaving model configuration to: {config_path}")
with open(config_path, "w") as f:
    cfg_dict = cfg.dump()
    f.write(cfg_dict)
print("Configuration saved successfully!")
```

## 12. Inference Function

```python
def run_inference(image_path, predictor, confidence_threshold=0.7):
    """
    Run inference on a single image and visualize results.
    """
    print(f"\nRunning inference on: {image_path}")
    
    # Read image
    print("Loading image...")
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    # Run inference
    print("Running model inference...")
    outputs = predictor(im)
    
    # Prepare visualization
    print("Preparing visualization...")
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
        print("Detections:")
        for cls, score in zip(classes, scores):
            print(f"- {cls}: {score:.3f}")
    
    return outputs, out.get_image()[:, :, ::-1]
```

## 13. Save Results Summary

```python
# Save a summary of the model and results
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

print(f"\nSaved model summary to: {summary_path}")
