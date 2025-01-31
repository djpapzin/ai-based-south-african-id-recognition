# South African ID Document Processing Project

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
- Total Images: 101
  * Training Set: 80 images
  * Validation Set: 21 images
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

- Phase 2 (Current):
  * Field detection model training
  * Using Detectron2 for object detection
  * Building end-to-end demo pipeline

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
   
2. 🔄 Field Detection (Current)
   - Training on 101 images
   - Building demo pipeline
   
3. 📋 Dataset Expansion
   - Target: 500+ additional images
   - Options:
     a) Model-assisted labeling (preferred)
     b) Manual labeling (fallback, 2-3 days)
   
4. 🎯 Final Implementation
   - Retrain with expanded dataset
   - OCR integration
   - Results generation

## 6. Next Steps
1. Complete current field detection training
2. Build end-to-end demo pipeline
3. Test model-assisted labeling in Label Studio
4. Based on accuracy:
   - If good: Proceed with auto-labeling
   - If poor: Switch to manual labeling
5. Implement OCR and complete pipeline

## 7. Open Questions

### Technical
1. Field detection model accuracy targets?
2. OCR confidence threshold and validation rules?
3. Label Studio integration approach?

### Process
1. Quality control measures for auto-labeling
2. Error handling and reporting
3. Data privacy compliance measures

## 8. Pending Items
1. Field detection model training completion
2. Demo pipeline implementation
3. Label Studio integration setup
4. OCR implementation plan
