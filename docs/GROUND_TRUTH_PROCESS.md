# Ground Truth Generation Process for SA ID Recognition

## Overview
This document outlines the process of generating ground truth data for the South African ID Recognition system using Large Language Models (LLMs).

## Status: ✓ COMPLETE

## Test Dataset
- Total Images: 100
  * New IDs: 50 images
  * Old IDs: 50 images
- Location: `test_dataset/`
  * New IDs: `test_dataset/new_ids/`
  * Old IDs: `test_dataset/old_ids/`

## Dataset Overview
- **Total Images Processed**: 96
  * New IDs: 48 unique images
  * Old IDs: 48 unique images
- **Location**: `ground_truth/raw_llm_responses/`
  * New IDs: `new_ids/results.json`
  * Old IDs: `old_ids/results.json`

## LLM Prompt Template
```
Please extract text from this South African ID document image and provide the results in JSON format. Include all visible text fields with exact formatting as shown in the image.

Required format:
{
    "id_type": "new|old",
    "id_number": "",
    "surname": "",
    "names": "",
    "nationality": "",
    "country_of_birth": "",
    "date_of_birth": "",  // Preserve original format (text or numeric month)
    "sex": "",
    "citizenship_status": ""
}

Important instructions:
1. Maintain EXACT character accuracy including spaces and formatting
2. For dates, preserve the original format (whether month is in text or numbers)
3. Include ONLY text that is clearly visible in the image
4. If a field is not visible or unclear, use null
5. Do not make assumptions or fill in missing data
```

## Process Steps
1. Image Preparation
   - [ ] Convert images to appropriate format for LLM
   - [ ] Ensure proper resolution and clarity
   - [ ] Organize in batches for processing

2. LLM Processing
   - [ ] Submit images in batches
   - [ ] Store raw LLM responses
   - [ ] Validate JSON format

3. Ground Truth Compilation
   - [ ] Collect all LLM responses
   - [ ] Validate data completeness
   - [ ] Format consistently
   - [ ] Store in structured format

4. Quality Control
   - [ ] Manual verification of subset
   - [ ] Check for formatting consistency
   - [ ] Validate against known patterns
   - [ ] Document any discrepancies

## Data Quality Metrics
1. Image Quality Distribution:
   - Clear: 90 images
   - Partial: 4 images
   - Poor: 2 images

2. Field Completeness:
   - ID Numbers: 100%
   - Names & Surnames: 100%
   - Dates: 100%
   - Citizenship: 100%

3. Format Variations:
   - ID Numbers:
     * New: Continuous (e.g., "0003011014087")
     * Old: Spaced (e.g., "650228 5970 08 2")
   - Dates:
     * New: "DD MMM YYYY" (e.g., "01 MAR 2000")
     * Old: "YYYY-MM-DD" (e.g., "1965-02-28")
   - Citizenship:
     * New: "CITIZEN"
     * Old: "S.A.BURGER/S.A.CITIZEN" or "S.A.CITIZEN"
   - Country of Birth:
     * Mixed: "SOUTH AFRICA" / "SUID-AFRIKA"
     * One instance of "CISKEI"

## Output Format
```json
{
    "image_id": "filename.jpg",
    "ground_truth": {
        "id_type": "new|old",
        "id_number": "",
        "surname": "",
        "names": "",
        "nationality": "",
        "country_of_birth": "",
        "date_of_birth": "",
        "sex": "",
        "citizenship_status": ""
    },
    "metadata": {
        "processing_date": "",
        "llm_model": "",
        "confidence_score": null
    }
}
```

## Data Storage Structure
```
ground_truth/
├── raw_llm_responses/
│   ├── new_ids/
│   │   └── results.json
│   └── old_ids/
│       └── results.json
└── validation/
    └── discrepancies.log
```

## Key Findings
1. Format Consistency:
   - Each ID type maintains its own consistent format
   - Date formats differ between old and new IDs
   - Citizenship status has variations in old IDs

2. Language Variations:
   - Bilingual entries (English/Afrikaans) in country of birth
   - Consistent English usage in new IDs
   - Mixed language usage in old IDs

3. Data Quality:
   - High overall quality (93.75% clear images)
   - Main uncertainties in signature visibility
   - Few cases of unclear text in old IDs

## Next Steps
1. Evaluation
   - [ ] Compare OCR results against ground truth
   - [ ] Calculate per-field accuracy metrics
   - [ ] Generate error analysis report

2. Documentation
   - [ ] Create field-specific validation rules
   - [ ] Document format variations for handling
   - [ ] Update OCR post-processing guidelines

3. Optimization
   - [ ] Enhance date format handling
   - [ ] Improve bilingual text processing
   - [ ] Optimize ID number formatting

## Search Tags
#ground-truth #data-quality #ocr-validation #south-african-id #document-processing #data-collection
