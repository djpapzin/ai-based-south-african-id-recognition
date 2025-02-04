# South African ID Document Processing Project

## Current Status (February 4, 2025)

### Dataset
- Total Images: 66
  * Training Set: 53 images
  * Validation Set: 13 images
- Image Format: Variable size (preserving aspect ratio)
- Annotation Format: COCO JSON with keypoints
- Categories: 15 total (11 fields + 4 keypoints)

### Dataset Processing Completed
1. Dataset Preparation
   - Unified annotation format with keypoints
   - Preserved image aspect ratios
   - Added corner keypoint annotations
   - Validated all annotations

2. Data Preparation
   - Split into train/val sets
   - Added keypoint metadata
   - Configured for Keypoint R-CNN
   - Ready for training

### Category System
1. Field Categories (Bounding Boxes):
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

2. Corner Points (Keypoints):
   - top_left_corner
   - top_right_corner
   - bottom_left_corner
   - bottom_right_corner

### Model Architecture
1. Base Model
   - Keypoint R-CNN with ResNet50-FPN backbone
   - Configured for both detection and keypoint estimation
   - Custom keypoint head for corner detection

2. Training Configuration
   - Batch size: 2 (GPU) / 1 (CPU)
   - Learning rate: 0.00025 with decay
   - 5000 iterations with evaluation every 500
   - 8 conv layers in keypoint head

### Technical Approach
1. Model Architecture
   - Framework: Detectron2
   - Environment: Google Colab (GPU)
   - Input Size: Variable size (preserving aspect ratio)
   - Output: Both bounding boxes and keypoints

2. Training Strategy
   - Use pre-trained weights
   - Train on standardized images
   - Validate on separate test set
   - Monitor both bbox and keypoint accuracy

### Next Steps
1. Model Training
   - Upload dataset to Google Drive
   - Set up Colab environment
   - Train initial model
   - Evaluate performance

2. Pipeline Development
   - Integrate with classification model
   - Add field detection
   - Implement OCR
   - Create demo interface

### Requirements
1. Performance
   - Classification Accuracy: 99%
   - Field Detection: High precision
   - Processing Time: <10 seconds
   - Format: JSON output

2. Features
   - Document type detection
   - Field extraction
   - Corner point detection
   - Text recognition
   - Confidence scores

### Notes
- Dataset is properly organized and validated
- All images are standardized
- Corner points included for better accuracy
- Category system handles both old and new IDs
- Ready for model training phase

## 1. Project Overview
The goal is to develop an AI system for recognizing and extracting metadata from South African ID documents:
- Traditional green bar-coded ID book
- New smart ID card

### Key Requirements
- 99% accuracy in document type classification
- 1% maximum error in metadata extraction
- 10-second maximum processing time per document
- POPIA compliance for data handling

## 2. Current Progress

### Completed Step 1: Document Classification
- Successfully trained classification model on 600+ images
- Labels: New ID vs Old ID document types
- Model performing as expected
- Provides foundation for field detection

### Current Step 2: Field Detection
#### Current Dataset
- Total Images: 66
  * Training Set: 53 images
  * Validation Set: 13 images
- Label Characteristics:
  * 12 labels per image (fields within ID)
  * No duplicate labels
  * Includes both rectangle and keypoint labels
- Status: Initial training in progress

### Technical Progress
- Phase 1 (Completed):
  * Successful document classification model (600+ images)
  * OCR baseline testing (Tesseract, EasyOCR, PaddleOCR)
  * Determined pure OCR approach insufficient
  * Research findings:
    - Separate models (classification + detection) recommended for accuracy
    - Processing speed can be optimized later
    - Normalization required for precise corner detection

- Phase 2 (Current):
  * Field detection model training
  * Using Detectron2 for object detection
  * Enhanced labeling approach:
    - Corner point detection for normalization
    - Consistent field naming convention established
    - Initial timing estimate: ~22 images per day

### Labeling Process Improvements
- Enhanced Annotation Strategy:
  * Corner points for document normalization
  * Standardized field names across team
  * More precise than simple bounding boxes
  * Coordination between team members for consistency

### Time Estimates
- Current labeling speed: ~22 images per day
- Manual labeling estimate (500+ images): 2-3 days
- Team coordination:
  * Abenathi joining for labeling support
  * Standardized process documentation
  * Quality control measures

## 3. Technical Implementation

### Current Focus: Field Detection
- Model: Detectron2 with ResNet50-FPN backbone
- Purpose: 
  * Field location detection (surname, name, DOB, etc.)
  * Corner point detection for ROI
- Status: Training in progress

### Planned Pipeline
1. Object Detection (Current)
   - Detect and classify all fields
   - Identify corner points

2. Label Studio Integration
   - Use trained model to pre-label 500+ additional images
   - Human verification workflow
   - Fallback to manual labeling if accuracy insufficient (2-3 days)

3. Model Refinement
   - Retrain with expanded dataset
   - Implement OCR for text extraction
   - Generate final results

## 4. Critical Requirements

### High Priority Fields
1. ID Number
2. Photo/Face
3. Names
4. Date of Birth

### Technical Requirements
- Support for common image formats (JPEG, PNG, TIFF)
- Error handling for poor quality images
- Data augmentation implementation
- POPIA compliance

## 5. Project Phases
1. ✓ Document Classification (Completed)
   - 600+ images labeled
   - Model trained and verified
   - Simple bounding box approach
   
2. 🔄 Field Detection (Current)
   - Training on 66 images
   - Enhanced labeling with corner points
   - Standardized field naming
   - Building demo pipeline
   
3. 📋 Dataset Expansion
   - Target: 500+ additional images
   - Options:
     a) Model-assisted labeling (preferred)
     b) Manual labeling (fallback, 2-3 days)
   - Quality control process established
   
4. 🎯 Final Implementation
   - Retrain with expanded dataset
   - OCR integration
   - Results generation
   - Speed optimization if needed

## 6. Next Steps
1. Complete current field detection training
2. Build end-to-end demo pipeline
3. Test model-assisted labeling in Label Studio
4. Based on accuracy:
   - If good: Proceed with auto-labeling
   - If poor: Switch to manual labeling (2-3 days)
5. Implement OCR and complete pipeline

## 7. Open Questions

### Technical
1. Field detection model accuracy targets?
2. OCR confidence threshold and validation rules?
3. Label Studio integration approach?
4. Speed optimization strategies if needed?

### Process
1. Quality control measures for auto-labeling
2. Error handling and reporting
3. Data privacy compliance measures
4. Team coordination procedures

## 8. Pending Items
1. Field detection model training completion
2. Demo pipeline implementation
3. Label Studio integration setup
4. OCR implementation plan
5. Standardized labeling documentation
6. Team training on labeling process
