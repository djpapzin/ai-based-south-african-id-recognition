# South African ID Card Information Extraction

A comprehensive solution for extracting and processing information from South African ID cards using Detectron2 for object detection.

## Running the Scripts

### Environment Setup
Before running any scripts, make sure to activate the correct conda environment:

```bash
conda activate detectron2_env
```

Alternatively, you can use the full path to the Python executable:
```bash
C:\Users\lfana\anaconda3\envs\detectron2_env\python.exe run_batch_inference.py
```

### Running Batch Inference
The `run_batch_inference.py` script processes multiple ID images and generates structured output. To run:

1. Activate the environment (as shown above)
2. Place your ID images in the appropriate directory:
   - New IDs: `test_dataset/new_ids/`
   - Old IDs: `test_dataset/old_ids/`
3. Run the script:
   ```bash
   python run_batch_inference.py
   ```

The script will:
- Process all images in both directories
- Generate OCR results in JSON format
- Create text summaries for each ID
- Show progress with a progress bar

## Current Status (February 7, 2025)

### Development Environments
1. **Google Colab (Training Environment)**
   - Used for model training and experimentation
   - GPU acceleration available
   - Jupyter notebook interface
   - Shared model storage via Google Drive

2. **Local Windows (Inference Environment)**
   - Used for production inference
   - CPU-based processing
   - Command-line interface
   - Local file system storage

### Object Detection
- Model: Detectron2 Faster R-CNN
- Performance:
  * AP (IoU=0.50:0.95): 52.30%
  * AP50 (IoU=0.50): 89.64%
  * AP75 (IoU=0.75): 53.40%

### OCR Pipeline
- Implemented Features:
  * Dual OCR Engine Support (Tesseract + PaddleOCR)
  * Field-specific preprocessing
  * OCR visualization
  * Results packaging (JSON/Text)
  * Image-only field handling

### Latest Updates
- Added local inference script with dual OCR support
- Implemented comprehensive error handling
- Created detailed setup and usage documentation
- Added field-specific OCR configurations
- Improved text cleaning for different field types

### Documentation Structure
```
project/
├── docs/                           # Project documentation
│   ├── SETUP_AND_USAGE_GUIDE.md    # Detailed setup instructions
│   ├── PROJECT_STATUS.md           # Current project status
│   ├── Checklist.md               # Development checklist
│   └── plan.md                    # Project planning
├── models/                         # Trained models
├── test_images/                    # Test image directory
└── outputs/                        # Processing outputs
    ├── classified/                # JSON results
    └── text_results/              # Text format results
```

## Setup Instructions

### Local Environment Setup
See `docs/SETUP_AND_USAGE_GUIDE.md` for detailed instructions on:
- Creating the conda environment
- Installing dependencies
- Setting up Tesseract OCR
- Running the inference script

### Google Colab Setup
1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Install Dependencies:
```python
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Directory Structure
```
project/
├── detected_segments/
│   └── image_name/
│       ├── face.jpg
│       ├── id_number.jpg
│       ├── signature.jpg
│       └── detection_metadata.json
├── model_output/
│   ├── model_final.pth
│   ├── model_cfg.yaml
│   └── metadata.json
└── dj_dataset/
├── train/
    │   ├── annotations.json
│   └── images/
└── val/
        ├── annotations.json
    └── images/
```

## Features

### 1. Field Detection
- 15 field categories including:
  * ID Document
  * Face
  * Signature
  * Personal Information Fields

### 2. Output Generation
- Segmented field images
- JSON metadata with:
  * Confidence scores
  * Bounding boxes
  * Field classifications

### 3. Processing Options
- Single image processing
- Batch processing
- Random sampling
- Confidence thresholding

## Next Steps
1. 🔄 Add PaddleOCR support
2. 🔄 Enhance OCR accuracy
3. 🔄 Improve visualization
4. 🔄 Add confidence scoring

## License

MIT License

Copyright (c) 2024 Kwantu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributors

### Core Team
- **DJ Papzin** - *Project Lead & ML Engineer*
  - Model architecture design
  - Training pipeline development
  - OCR integration

- **Abenathi** - *ML Engineer*
  - Dataset preparation
  - Model training
  - Performance optimization

### Organization
- **Kwantu** - Project Sponsor & Support
  - Infrastructure support
  - Dataset provision
  - Project guidance

### Contact
For any queries regarding this project, please contact:
- Email: info@kwantu.com
- Website: [kwantu.com](https://kwantu.com)

## Project Components

### 1. Object Detection and Keypoint Model (Current Focus)
- Detectron2-based Keypoint R-CNN model for detecting key regions in ID cards:
  - ID Number
  - Names
  - Surname
  - Date of Birth
  - Sex
  - Face Photo
  - Type of ID
- Training Status: Completed
  - Base Model: Faster R-CNN with FPN backbone (pretrained on COCO)
  - Dataset Distribution:
    - Training Set: 8 images with 58 total instances
    - Validation Set: 2 images with 14 instances
    - Categories: 7 (all fields present in each image)
  - Training Progress:
    - Total Loss: Decreased from ~5.5 to ~0.56 (showing good convergence)
    - Classification Loss: Improved from ~1.9 to ~0.15
    - Box Regression Loss: Improved from ~0.86 to ~0.33
    - Learning Rate: Progressive increase from 5e-6 to 2e-4

- Evaluation Results:
  - Overall Performance:
    - mAP (IoU=0.50:0.95): 37.34%
    - AP50 (IoU=0.50): 79.89%
    - AP75 (IoU=0.75): 21.50%
  
  - Per-Category AP:
    - Type of ID: 100.00% (Best performing)
    - Date of Birth: 42.53%
    - Names: 35.05%
    - ID Number: 33.40%
    - Face Photo: 30.30%
    - Surname: 10.10%
    - Sex: 10.00% (Needs improvement)
  
  - Size-based Performance:
    - Small objects: AP = 24.00%
    - Large objects: AP = 42.90%
    - Medium objects: Not applicable in current dataset

- Architecture Details:
  - Backbone: ResNet50-FPN
  - ROI Heads: Box Predictor with 8 classes
  - Input Size: Dynamic (640-800px shortest edge)
  - Augmentations: ResizeShortestEdge, RandomFlip

### 2. OCR Processing
- Multiple OCR engine support:
  - Tesseract OCR
  - EasyOCR
  - PaddleOCR
- Features:
- Handles both images (.jpg, .jpeg, .png) and PDFs
  - Automatic rotation correction
- Region of Interest (ROI) extraction
  - Parallel processing
- Comprehensive preprocessing pipeline

## Setup Instructions

### Prerequisites
1. Python 3.7+
2. CUDA-capable GPU (for training)
3. Tesseract OCR
4. Poppler (for PDF processing)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Model Training (Detectron2)

1. Data Preparation:
```bash
# Split dataset into train/val sets
python split_coco_dataset.py --input annotations.json --output-dir pipeline_output
```

2. Training:
- Use Google Colab with GPU runtime
- Follow the notebook: `SA_ID_Book_Training_Detectron2.ipynb`

3. Training Monitoring:
```python
# Add this code block to your training notebook for real-time metrics visualization
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import os

def plot_training_metrics(metrics_file="/content/training_output/metrics.json"):
    metrics = defaultdict(list)
    
    # Read metrics from the JSON file
    with open(metrics_file) as f:
        for line in f:
            record = json.loads(line)
            for k, v in record.items():
                if "loss" in k or "lr" in k:
                    metrics[k].append(v)
            metrics["iteration"].append(record["iteration"])
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Metrics")
    
    # Plot total loss
    axes[0, 0].plot(metrics["iteration"], metrics["total_loss"])
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(True)
    
    # Plot component losses
    axes[0, 1].plot(metrics["iteration"], metrics["loss_cls"], label="Classification")
    axes[0, 1].plot(metrics["iteration"], metrics["loss_box_reg"], label="Box Regression")
    axes[0, 1].set_title("Component Losses")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot RPN losses
    axes[1, 0].plot(metrics["iteration"], metrics["loss_rpn_cls"], label="RPN Class")
    axes[1, 0].plot(metrics["iteration"], metrics["loss_rpn_loc"], label="RPN Location")
    axes[1, 0].set_title("RPN Losses")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot learning rate
    axes[1, 1].plot(metrics["iteration"], metrics["lr"])
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Monitor metrics every N iterations
def monitor_training():
    if os.path.exists("/content/training_output/metrics.json"):
        plot_training_metrics()
    else:
        print("Metrics file not found. Training may not have started yet.")
```

4. Training Duration and Checkpoints:
- Expected Duration:
  - Full Training: ~2-3 hours on Google Colab GPU
  - Current Progress: ~45 minutes per 1000 iterations
  - Total Iterations: 3000 (configurable)

- Checkpointing:
  - Location: `/content/training_output/`
  - Frequency: Every 1000 iterations
  - Files:
    - `model_final.pth`: Final trained model
    - `model_####.pth`: Intermediate checkpoints
    - `metrics.json`: Training metrics
    - `last_checkpoint`: Points to latest checkpoint

- Recovery from Interruption:
```python
# Add this code to resume training from last checkpoint
from detectron2.checkpoint import DetectionCheckpointer

# Resume training
cfg.MODEL.WEIGHTS = "/content/training_output/model_last.pth"  # Path to last checkpoint
trainer = Trainer(cfg)
DetectionCheckpointer(trainer.model).resume_or_load(cfg.MODEL.WEIGHTS)
trainer.resume_or_load()
```

## Project Structure

```
.
├── annotated_images/     # Original dataset with images
└── pipeline_output/      # Processed dataset for Colab
    ├── train.json       # Training annotations (created in Colab)
    └── val.json         # Validation annotations (created in Colab)
```

## Scripts Overview

### Google Colab Notebook
`SA_ID_Book_Training_Detectron2.ipynb`
- Complete end-to-end solution including:
  - Environment setup & dependency installation
  - Dataset loading and preprocessing
  - Dataset splitting (train/val)
  - Model configuration and training
  - Real-time metrics visualization
  - Inference and visualization
  - Model export and evaluation

Note: All operations (training, inference, visualization) are performed directly in the Colab notebook. No local scripts are required as the complete pipeline is integrated into the notebook for better reproducibility and ease of use.

## Workflow

1. **Data Preparation**:
   - Organize annotated images in `annotated_images/` directory
   - Upload to Google Drive

2. **Training (in Colab)**:
   - Open `SA_ID_Book_Training_Detectron2.ipynb` in Google Colab
   - Mount Google Drive
   - Run all cells sequentially
   - Monitor training progress using built-in visualizations

3. **Inference (in Colab)**:
   - Use the provided inference cells in the notebook
   - Visualize results directly in Colab
   - Export predictions as needed

## Current Status

- ✓ Data preparation completed
- ✓ Training infrastructure set up
- ✓ Model training completed
- ✓ Initial evaluation completed

## Next Steps

1. ~~Complete model training~~
2. ~~Evaluate model performance~~
3. Fine-tune model to improve:
   - Sex field detection (currently 10% AP)
   - Surname detection (currently 10.1% AP)
   - Small object detection (currently 24% AP)
4. Expand dataset size (currently using only 10 images)
5. Create inference pipeline
6. Document usage guidelines

## Testing the Model

### 1. Model Loading
```python
# Load the trained model for inference
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("/content/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # your number of classes
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/id_card_model/training_output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    return DefaultPredictor(cfg)

predictor = setup_predictor()
```

### 2. Running Inference
```python
def visualize_prediction(image_path):
    # Read image
    im = cv2.imread(image_path)
    
    # Make prediction
    outputs = predictor(im)
    
    # Get predictions
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    
    # Visualization
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Define class names
    class_names = ["Date of Birth", "Face Photo", "ID Number", "Names", "Sex", "Surname", "Type of ID"]
    
    # Draw predictions
    for box, class_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw box
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_names[class_id]}: {score:.2f}"
        cv2.putText(im, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display result
    plt.figure(figsize=(15,10))
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    
    # Print detections
    print("\nDetections:")
    for class_id, score in zip(classes, scores):
        print(f"{class_names[class_id]}: {score:.2f}")
```

### 3. Test on New Images
```python
# Test on a single image
test_image_path = "/path/to/your/test/image.jpg"  # Replace with your test image path
visualize_prediction(test_image_path)

# Test on multiple images
import glob
test_images = glob.glob("/path/to/test/images/*.jpg")  # Replace with your test images directory
for image_path in test_images:
    print(f"\nProcessing: {image_path}")
    visualize_prediction(image_path)
```

### 4. Interpreting Results
- Green boxes show detected regions
- Labels show class name and confidence score
- Higher confidence scores (closer to 1.0) indicate more certain predictions
- Threshold can be adjusted in `setup_predictor()` by changing `SCORE_THRESH_TEST`