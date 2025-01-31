# South African ID Book Detection Training with Detectron2

This notebook trains a Detectron2 model for detecting South African ID books using a pre-split dataset.
The notebook supports both GPU and CPU environments, with automatic device selection.

## 1. Install Dependencies

```python
import subprocess
import sys

def run_pip_install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install packages
run_pip_install('torch')
run_pip_install('torchvision')
run_pip_install('git+https://github.com/facebookresearch/detectron2.git')
```

## 2. Import Libraries

```python
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
```

## 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 4. GPU Check Function

```python
def check_gpu():
    """Check GPU availability and print device information."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    return torch.cuda.is_available()

# Check GPU
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
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    
    # Training configuration
    if use_gpu:
        cfg.SOLVER.IMS_PER_BATCH = 2
    else:
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.MODEL.DEVICE = "cpu"
    
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg
```

## 6. Dataset Verification Function

```python
def verify_dataset(json_path, images_dir):
    """Verify dataset integrity."""
    print(f"\nVerifying dataset at {json_path}")
    
    # Check JSON file
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                coco_data = json.load(f)
                print(f"✓ JSON file is valid")
                print(f"  - Number of images: {len(coco_data['images'])}")
                print(f"  - Number of annotations: {len(coco_data['annotations'])}")
                print(f"  - Categories: {[cat['name'] for cat in coco_data['categories']]}")
            except json.JSONDecodeError:
                print("✗ Error: Invalid JSON file")
                return False
    else:
        print("✗ Error: JSON file not found")
        return False
    
    # Check images directory
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ Found {len(image_files)} images")
        
        # Verify a random image
        if image_files:
            test_image = os.path.join(images_dir, random.choice(image_files))
            img = cv2.imread(test_image)
            if img is not None:
                print(f"✓ Successfully loaded test image")
            else:
                print("✗ Error: Failed to load test image")
                return False
    else:
        print("✗ Error: Images directory not found")
        return False
    
    return True
```

## 7. Visualization Functions

```python
def visualize_dataset(dataset_name, num_samples=3):
    """Visualize random samples from the dataset."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    
    # Get random samples
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    for d in samples:
        img = cv2.imread(d["file_name"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        
        if "annotations" in d:
            # Draw annotations
            vis = visualizer.draw_dataset_dict(d)
        else:
            vis = visualizer.draw_instance_predictions(d)
            
        plt.figure(figsize=(15, 10))
        plt.imshow(vis.get_image())
        plt.title(f"Sample from {dataset_name}\nFile: {os.path.basename(d['file_name'])}")
        plt.axis('off')
        plt.show()
        print(f"Image size: {img.shape}")
        print(f"Number of annotations: {len(d.get('annotations', []))}")
        print("Annotation types:", [ann.get('category_id', 'unknown') for ann in d.get('annotations', [])])
        print("-" * 50)

def visualize_prediction(predictor, image_path):
    """Visualize model prediction on an image."""
    im = cv2.imread(image_path)
    if im is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    outputs = predictor(im)
    
    # Convert BGR to RGB for visualization
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Create visualizer
    v = Visualizer(im,
                   metadata=MetadataCatalog.get("sa_id_val"),
                   scale=0.8)
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(15, 10))
    plt.imshow(out.get_image())
    plt.title(f"Predictions on {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()
    
    # Print prediction details
    print("\nPrediction Details:")
    print(f"Number of detections: {len(outputs['instances'])}")
    if len(outputs['instances']) > 0:
        print("Confidence scores:", outputs['instances'].scores.tolist())
        print("Predicted classes:", outputs['instances'].pred_classes.tolist())

# Test the visualization functions
print("Visualizing training dataset samples:")
visualize_dataset("sa_id_train", num_samples=2)

print("\nVisualizing validation dataset samples:")
visualize_dataset("sa_id_val", num_samples=2)
```

## 8. Dataset Annotation Fix Function

```python
import json
import cv2
import os
from tqdm import tqdm

def fix_dataset_annotations(json_path, images_dir):
    """Fix dataset annotations by updating width and height to match actual image dimensions."""
    print(f"\nFixing annotations in {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fixed_count = 0
    next_ann_id = 1  # Start with ID 1
    valid_annotations = []
    
    # First fix image dimensions
    for img in tqdm(data['images']):
        image_path = os.path.join(images_dir, img['file_name'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                actual_height, actual_width = image.shape[:2]
                if img['width'] != actual_width or img['height'] != actual_height:
                    print(f"\nFixing {img['file_name']}: {img['width']}x{img['height']} -> {actual_width}x{actual_height}")
                    img['width'] = actual_width
                    img['height'] = actual_height
                    fixed_count += 1
    
    # Fix annotations and ensure all required fields are present
    for ann in data['annotations']:
        # Check if annotation has all required fields
        if 'bbox' not in ann or 'category_id' not in ann or 'image_id' not in ann:
            continue
            
        # Ensure bbox is valid (x, y, width, height format)
        if len(ann['bbox']) != 4:
            continue
            
        # Ensure bbox values are valid numbers
        if not all(isinstance(x, (int, float)) for x in ann['bbox']):
            continue
            
        # Add required fields if missing
        ann['id'] = next_ann_id
        next_ann_id += 1
        ann['bbox_mode'] = 0  # XYWH_ABS
        ann['iscrowd'] = ann.get('iscrowd', 0)
        ann['area'] = float(ann['bbox'][2] * ann['bbox'][3])  # width * height
        
        valid_annotations.append(ann)
    
    # Update annotations with valid ones only
    data['annotations'] = valid_annotations
    
    # Ensure category IDs start from 1
    category_id_map = {cat['id']: idx + 1 for idx, cat in enumerate(data['categories'])}
    for cat in data['categories']:
        cat['id'] = category_id_map[cat['id']]
    
    # Update category IDs in annotations
    for ann in data['annotations']:
        ann['category_id'] = category_id_map[ann['category_id']]
    
    # Always save the fixed annotations to ensure IDs are unique
    output_path = json_path.replace('.json', '_fixed.json')
    with open(output_path, 'w') as f:
        json.dump(data, f)
    print(f"\nFixed {fixed_count} annotations. Saved to {output_path}")
    print(f"Total valid annotations: {len(valid_annotations)}")
    return output_path
```

## 9. Setup and Register Datasets

```python
# Set paths for your existing dataset structure
GDRIVE_PATH = "/content/drive/MyDrive/Kwantu/Machine Learning"
DATASET_PATH = os.path.join(GDRIVE_PATH, "merged_dataset")
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
OUTPUT_DIR = os.path.join(GDRIVE_PATH, "model_output")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fix annotations first
train_json = os.path.join(TRAIN_PATH, "annotations.json")
val_json = os.path.join(VAL_PATH, "annotations.json")
train_images = os.path.join(TRAIN_PATH, "images")
val_images = os.path.join(VAL_PATH, "images")

# Fix and get paths to fixed annotation files
fixed_train_json = fix_dataset_annotations(train_json, train_images)
fixed_val_json = fix_dataset_annotations(val_json, val_images)

# Clean up existing registrations
for d in ["sa_id_train", "sa_id_val"]:
    if d in DatasetCatalog:
        DatasetCatalog.remove(d)
    if d in MetadataCatalog:
        MetadataCatalog.remove(d)

# Register datasets with fixed annotations
register_coco_instances("sa_id_train", {}, fixed_train_json, train_images)
register_coco_instances("sa_id_val", {}, fixed_val_json, val_images)

# Verify registration
print("\nVerifying dataset registration:")
train_dicts = DatasetCatalog.get("sa_id_train")
val_dicts = DatasetCatalog.get("sa_id_val")
print(f"✓ Training dataset registered with {len(train_dicts)} images")
print(f"✓ Validation dataset registered with {len(val_dicts)} images")

# Print sample annotation to verify structure
if len(train_dicts) > 0:
    sample_dict = train_dicts[0]
    print("\nSample annotation structure:")
    if 'annotations' in sample_dict:
        print(f"Number of annotations: {len(sample_dict['annotations'])}")
        if len(sample_dict['annotations']) > 0:
            print("First annotation fields:", sample_dict['annotations'][0].keys())
```

## 10. Dataset Verification

```python
def verify_dataset_structure(dataset_dicts):
    """Verify the structure of dataset annotations."""
    print("\nVerifying dataset structure...")
    has_errors = False
    
    for idx, d in enumerate(dataset_dicts):
        # Check basic required fields
        required_fields = ['file_name', 'height', 'width', 'image_id']
        missing_fields = [field for field in required_fields if field not in d]
        if missing_fields:
            print(f"❌ Image {idx}: Missing fields: {missing_fields}")
            print(f"   File: {d.get('file_name', 'unknown')}")
            has_errors = True
            continue
        
        # Check annotations
        if 'annotations' not in d:
            print(f"❌ Image {idx}: No annotations found")
            print(f"   File: {d['file_name']}")
            has_errors = True
            continue
            
        for ann_idx, ann in enumerate(d['annotations']):
            # Print full annotation for debugging
            print(f"\nChecking annotation {ann_idx} in image {idx}")
            print(f"Annotation contents: {ann}")
            
            # Check annotation required fields
            ann_required_fields = ['bbox', 'bbox_mode', 'category_id']
            missing_ann_fields = [field for field in ann_required_fields if field not in ann]
            
            if missing_ann_fields:
                print(f"❌ Image {idx}, Annotation {ann_idx}: Missing fields: {missing_ann_fields}")
                print(f"   File: {d['file_name']}")
                has_errors = True
                continue
            
            # Verify bbox format if present
            if 'bbox' in ann:
                try:
                    if len(ann['bbox']) != 4:
                        print(f"❌ Image {idx}, Annotation {ann_idx}: Invalid bbox format")
                        print(f"   Found bbox: {ann['bbox']}")
                        print(f"   File: {d['file_name']}")
                        has_errors = True
                except Exception as e:
                    print(f"❌ Image {idx}, Annotation {ann_idx}: Error processing bbox")
                    print(f"   Error: {str(e)}")
                    print(f"   Bbox value: {ann.get('bbox', 'Not found')}")
                    print(f"   File: {d['file_name']}")
                    has_errors = True
    
    if has_errors:
        print("\n⚠️ Dataset verification found errors that need to be fixed")
    else:
        print("✓ Dataset structure verification complete - No errors found")
    
    return not has_errors

# Verify both datasets
print("\nVerifying training dataset:")
train_ok = verify_dataset_structure(train_dicts)
print("\nVerifying validation dataset:")
val_ok = verify_dataset_structure(val_dicts)

if not (train_ok and val_ok):
    print("\n⚠️ Please fix the dataset errors before proceeding with training")
else:
    print("\n✓ All dataset checks passed - Ready for training")
```

## 11. Training with TensorBoard

```python
# Setup TensorBoard
import datetime
%load_ext tensorboard
from torch.utils.tensorboard import SummaryWriter

# Create a unique log directory
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(OUTPUT_DIR, 'logs', current_time)
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Clear any existing TensorBoard instances
!kill -9 $(lsof -t -i:6006) 2>/dev/null || true
%tensorboard --logdir={log_dir}

# Configure training
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("sa_id_train",)
cfg.DATASETS.TEST = ()

# Set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {cfg.MODEL.DEVICE}")

# Model parameters
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Update with your number of classes
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

# Training parameters
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.STEPS = []
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 500

# Output directory
cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("\nTraining configuration:")
print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
print(f"ROI batch size per image: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}")
print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")

# Create custom trainer to log metrics to TensorBoard
class TensorboardTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def after_step(self):
        # Log metrics after each step
        metrics = self.storage.latest()
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, self.iter)
        super().after_step()

# Initialize trainer
trainer = TensorboardTrainer(cfg)
print("\nStarting training...")
trainer.resume_or_load(resume=False)
trainer.train()

# Close TensorBoard writer
writer.close()
```

## 12. Save and Export Model

```python
# Save model configuration
final_model_path = os.path.join(OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = final_model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set detection threshold

# Save configuration
config_path = os.path.join(OUTPUT_DIR, "model_config.yaml")
with open(config_path, "w") as f:
    f.write(cfg.dump())
print(f"Saved model configuration to: {config_path}")

# Create predictor for inference
predictor = DefaultPredictor(cfg)
```

## 13. Running Inference (Standalone)

This section can be run in a new notebook to perform inference using the saved model.

```python
# Install dependencies if not already installed
import subprocess
import sys

def run_pip_install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install required packages
required_packages = [
    'torch',
    'torchvision',
    'git+https://github.com/facebookresearch/detectron2.git',
    'opencv-python',
    'matplotlib'
]

for package in required_packages:
    run_pip_install(package)

# Import required libraries
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set paths using the same structure as training
GDRIVE_PATH = "/content/drive/MyDrive/Kwantu/Machine Learning"
MODEL_PATH = os.path.join(GDRIVE_PATH, "model_output/model_final.pth")
CONFIG_PATH = os.path.join(GDRIVE_PATH, "model_output/model_config.yaml")

# Class names for SA ID detection
CLASS_NAMES = [
    "id_number", "surname", "names", "nationality",
    "country_of_birth", "status", "sex", "date_of_birth",
    "id_number_barcode", "identity_number_back",
    "control_number", "district_of_birth", "issue_date",
    "id_photo", "id_book"
]

def setup_model(model_path, config_path, confidence_threshold=0.7):
    """
    Setup the Detectron2 model for inference.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Load custom configuration if available
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    
    # Set model weights
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Register metadata
    MetadataCatalog.get("sa_id_test").set(thing_classes=CLASS_NAMES)
    
    return DefaultPredictor(cfg)

def run_inference(image_path, predictor, confidence_threshold=0.7):
    """
    Run inference on a single image and visualize results.
    Returns the predictions and visualization.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Run inference
    outputs = predictor(image)
    
    # Get predictions
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    # Get metadata for visualization
    metadata = MetadataCatalog.get("sa_id_test")
    
    # Create visualization
    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata,
                   scale=1.0)
    
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    vis_image = out.get_image()[:, :, ::-1]
    
    # Prepare results
    results = []
    for box, score, class_id in zip(boxes, scores, classes):
        if score >= confidence_threshold:
            results.append({
                "class": CLASS_NAMES[class_id],
                "confidence": float(score),
                "bbox": box.tolist()
            })
    
    return results, vis_image

def display_results(results, vis_image, image_path):
    """
    Display the results and visualization.
    """
    # Plot the image with detections
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Detections for {os.path.basename(image_path)}")
    plt.show()
    
    # Print detection results
    print(f"\nDetections for {os.path.basename(image_path)}:")
    for result in sorted(results, key=lambda x: x['confidence'], reverse=True):
        print(f"{result['class']}: {result['confidence']:.2f}")

def process_images(image_paths, model_path, config_path, confidence_threshold=0.7):
    """
    Process multiple images using the model.
    """
    # Setup model
    print(f"Setting up model from {model_path}")
    predictor = setup_model(model_path, config_path, confidence_threshold)
    print(f"Using device: {predictor.model.device}")
    
    # Process each image
    for image_path in image_paths:
        try:
            results, vis_image = run_inference(image_path, predictor, confidence_threshold)
            display_results(results, vis_image, image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Use validation images path from training setup
    VAL_PATH = os.path.join(GDRIVE_PATH, "merged_dataset/val")
    VAL_IMAGES_DIR = os.path.join(VAL_PATH, "images")
    
    print(f"\nUsing validation images from: {VAL_IMAGES_DIR}")
    
    # Get list of validation images
    image_paths = [
        os.path.join(VAL_IMAGES_DIR, f) 
        for f in os.listdir(VAL_IMAGES_DIR) 
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_paths:
        print(f"No validation images found in {VAL_IMAGES_DIR}")
        print("Please check that your validation dataset is correctly set up.")
    else:
        print(f"Found {len(image_paths)} validation images")
        # Run inference
        process_images(image_paths, MODEL_PATH, CONFIG_PATH, confidence_threshold=0.7)
```

## 14. Save Results Summary

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
