# South African ID Recognition Project Plan

## Project Overview
An automated system for detecting and extracting information from South African ID documents using computer vision and OCR technologies.

## Project Status Overview

### Completed Tasks:
1. Dataset Preparation
   - Successfully merged Abenathi's and DJ's datasets
   - Total of 101 fully labeled images
   - Split into 80 training and 21 validation images
   - All images standardized to 800x800 pixels
   - Unified category system for both old and new IDs

2. Data Processing
   - Implemented corner point detection
   - Standardized image formats and sizes
   - Validated all annotations
   - Created clean train/val split
   - Prepared dataset for Colab training

### Current Phase:
- Ready for model training with:
  * Training set: 80 images
  * Validation set: 21 images
  * 15 unified categories including:
    - 11 field categories (id_document, surname, names, etc.)
    - 4 corner point categories
  * Both bounding boxes and keypoints

## Implementation Strategy

### Immediate Next Steps:
1. Model Training (Priority)
   - Upload train_val_dataset.zip to Google Drive
   - Set up Colab environment
   - Train initial model
   - Evaluate performance

2. Pipeline Development
   - Integrate classification model
   - Add field detection
   - Implement OCR
   - Create JSON output structure

### Short-term Goals (This Week):
1. Complete model training
2. Evaluate model performance
3. Begin pipeline integration
4. Test on new images

### Medium-term Goals:
1. Complete end-to-end pipeline
2. Add error handling
3. Implement confidence scores
4. Create demo interface

## Technical Stack
- Object Detection: Detectron2
- Training Environment: Google Colab
- Image Processing: OpenCV
- Data Format: COCO JSON
- OCR Engine: Tesseract v5

## Dataset Structure
```
train_val_dataset.zip
├── train/
│   ├── annotations.json (80 images)
│   └── images/
└── val/
    ├── annotations.json (21 images)
    └── images/
```

## Categories
1. Bounding Box Categories:
   - id_document
   - surname
   - names
   - sex (New ID only)
   - nationality (New ID only)
   - id_number
   - date_of_birth
   - country_of_birth
   - citizenship_status
   - face
   - signature (New ID only)

2. Keypoint Categories:
   - top_left_corner
   - top_right_corner
   - bottom_left_corner
   - bottom_right_corner

## Performance Requirements
- Document Classification: 99% accuracy
- Field Detection: High precision using corner points
- Processing Time: <10 seconds total
- Output: Normalized, accurate field coordinates
- Format: Structured JSON with confidence scores

## Next Steps
1. Set up Colab training environment
2. Train initial model
3. Evaluate performance
4. Begin pipeline integration
5. Implement confidence scoring
6. Create demo interface

## Notes
- Dataset is now properly organized and validated
- All images are standardized to 800x800
- Corner points are included for better accuracy
- Category system handles both old and new IDs
