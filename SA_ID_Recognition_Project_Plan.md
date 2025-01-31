# South African ID Recognition Project Plan

## Project Overview
An automated system for detecting and extracting information from South African ID documents using computer vision and OCR technologies.

## Project Status Overview

### Completed (Step 1):
- Classification Model
  * Dataset: 600+ labeled images (New ID vs Old ID)
  * Model trained and validated
  * Working as expected

### Current Phase (Step 2):
- Object Detection Model
  * Dataset: 101 images fully labeled with all fields
  * Initial training results:
    - Overall Accuracy (AP50): 91%
    - Strong: ID document (88.9%), face (72.4%), DOB (65.9%)
    - Needs improvement: Sex (33.4%), Signature (34.4%)

## Technical Decisions

### Model Architecture
- Separate models for classification and detection
  * Reason: Optimizes for accuracy (>99% requirement)
  * Trade-off: Slightly longer processing time, but still within 10s requirement

### Document Detection Approach
- Enhanced detection using corner points
  * Adding corner point labeling for precise document location
  * Enables proper image normalization
  * Improves accuracy of field extraction

### Field Labeling Standards
- Consistent naming convention established
- Required fields defined for both document types
- Normalized coordinate system for better accuracy

## Implementation Strategy

### Current Focus (Weekend Development):
1. End-to-End Demo Pipeline
   - Input: ID document image
   - Classification: Use existing model (Old/New ID)
   - Field Detection: Current Detectron2 model
   - OCR Integration: Tesseract
   - Output: JSON + extracted fields

### Weekend Development Plan

1. Saturday (Feb 1):
   - Pipeline Integration
     * Classification model integration
     * Detectron2 field detection
     * Corner point processing
     * Field normalization
     * Basic OCR implementation

2. Sunday (Feb 2):
   - Pipeline Refinement
     * JSON output structure
     * Result organization
     * Error handling
     * Basic interface
   - Testing and Documentation

3. Monday (Feb 3):
   - Morning: Final testing
   - Demo presentation
   - Feedback collection

### Next Phase:
1. Dataset Expansion
   - Use current model for pre-labeling (if accurate enough)
   - Target: 500+ labeled images
   - Manual verification/correction
2. Model Improvement
   - Retrain with expanded dataset
   - Focus on lower-performing fields
3. Pipeline Refinement
   - Optimize OCR accuracy
   - Improve processing speed
   - Enhanced error handling

## Technical Requirements
- Document Classification: 99% accuracy
- Field Detection: High precision using corner points
- Processing Time: <10 seconds total
- Output: Normalized, accurate field coordinates
- Format: Structured JSON with confidence scores

## Technical Stack
- Document Classification: Existing trained model
- Object Detection: Detectron2
- OCR Engine: Tesseract v5
- Image Processing: OpenCV
- Data Format: JSON

## Immediate Deliverables (By Monday)
1. Working demo pipeline showing:
   - Document type classification
   - Field detection and extraction
   - OCR text extraction
   - Organized output (JSON + images)
2. Basic interface for demo
3. Documentation of current capabilities

## Performance Requirements
- Document Classification: 99% accuracy
- Metadata Extraction: ≤ 1% error rate
- Processing Time: ≤ 10 seconds/document
- Format Support: JPEG, PNG, TIFF

## Team Structure
- Technical Implementation Lead: DJ
- Labeling Team: DJ and Abenathi
- Project Oversight: Willem and Robert
- Focus on labeling accuracy and verification

## Project Requirements

### Performance Metrics
- Document Classification Accuracy: 99%
- Metadata Extraction Error Rate: ≤ 1%
- Processing Time: ≤ 10 seconds per document
- Support for Common Formats: JPEG, PNG, TIFF

### Data Processing Pipeline
1. Document Classification
   - Distinguish between traditional ID book and smart ID card
   - ROI extraction with corner points
2. Field Detection
   - 12 labeled fields per document
   - Accurate bounding box detection
3. Text Extraction
   - OCR on detected fields
   - Format validation (especially ID numbers)
4. Special Field Processing
   - Face photo extraction
   - Signature extraction
5. Output Generation
   - JSON format
   - Confidence scores
   - Extracted text and images

## Timeline and Milestones

### Week 1 (Completed)
- Dataset collection and annotation
- 101 images fully labeled
- Data verification and quality checks

### Week 2-3 (In Progress)
- Model training and optimization
- Integration with OCR
- Performance evaluation

### Week 4 (Upcoming)
- Field validation implementation
- System integration
- Documentation
- Performance benchmarking

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
