# South African ID Recognition Project Plan

## Project Overview
An automated system for detecting and extracting information from South African ID documents using computer vision and OCR technologies.

## Current Status
- Training dataset: 100 labeled images
- Unlabeled dataset: 900+ images
- Target dataset size: 500 images (minimum for demo)
- Currently training Detectron2 model for field detection

## Implementation Phases

### Phase 1: Field Detection with Detectron2
- **Current Stage**: Training model on 100 labeled images
- **Purpose**: Detect and localize key fields in ID documents
- **Key Fields**:
  - ID Number (High Priority)
  - Photo (High Priority)
  - Names (High Priority)
  - Other fields (Secondary Priority)
- **Goal**: Use trained model for pre-labeling remaining dataset

### Phase 2: OCR Integration
- Using existing Tesseract OCR script
- Planned improvements:
  - Pre-processing optimizations
  - Confidence score calculation (percentage-based)
- Runs as a separate step after field detection

### Phase 3: Data Processing Pipeline
1. **Input**: Raw ID document image
2. **Field Detection**: Detectron2 model
   - Output: Bounding boxes for each field
   - Include confidence scores
3. **Image Processing**:
   - Crop detected regions
   - Store photo and signature as images
4. **Text Extraction**:
   - Apply OCR on text fields
   - Validate ID number format
5. **Output**: JSON format including
   - Extracted text
   - Field locations
   - Confidence scores (both detection and OCR)
   - Links to cropped images (photo/signature)

## Backup Plan (If Needed)
- Integration with Azure OpenAI (GPT-4V)
- Pending privacy clearance from Rob and Willem
- Would be used for pre-labeling assistance

## Success Criteria
1. Accurate field detection (emphasis on ID number, photo, names)
2. Reliable OCR extraction
3. Proper validation of ID number format
4. Complete JSON output with confidence scores
5. Successfully labeled dataset of 500+ images

## Next Steps
1. Complete current Detectron2 model training
2. Evaluate model performance on test set
3. Begin pre-labeling process on unlabeled dataset
4. Integrate with existing OCR pipeline
5. Implement confidence scoring system
6. Create final JSON output format

## Notes
- Existing OCR script needs minor improvements
- Focus on Detectron2 approach before considering GPT-4V integration
- Target of 500 images set for initial demo purposes
- System should include confidence scores in percentage format
