# OCR ID Number Extractor

This project evaluates different OCR (Optical Character Recognition) engines for extracting ID numbers from South African ID documents. The script processes images of ID documents and compares the performance of multiple OCR engines in terms of accuracy, speed, and reliability.

## Features

- Supports multiple OCR engines:
  - Tesseract OCR (fastest, requires local installation)
  - EasyOCR (good at partial matches)
  - PaddleOCR (highest accuracy for complete matches)
- Parallel image processing for improved performance
- Comprehensive evaluation metrics
- Detailed reporting in both JSON and Markdown formats
- Robust error handling and logging
- Image preprocessing optimized for ID documents

## Project Structure

```
.
├── IDcopies/              # Place ID images here in folders named with the ID number
│   └── .gitkeep          # Keeps the empty directory in git
├── results/              # Generated results and reports
│   ├── ocr_results.json  # Detailed JSON results
│   └── ocr_evaluation_report.md  # Summary report in markdown
├── ocr_evaluator.py     # Main OCR evaluation script
└── requirements.txt     # Python dependencies
```

## Directory Structure for ID Images

Place your ID images in the `IDcopies` directory using this structure:
```
IDcopies/
    ├── 9001015800085/  # Folder named with the ground truth ID number
    │   ├── image1.jpg
    │   └── image2.jpg
    └── 8505015800085/  # Another folder with different ID number
        ├── image3.jpg
        └── image4.jpg
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Tesseract OCR:
   - Windows: Download from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

## Usage

1. Place your ID images in the `IDcopies` directory following the structure above
2. Run the script:
```bash
python ocr_evaluator.py
```

The script will:
- Process images from the first 10 ID folders
- Generate a detailed evaluation report
- Save results in the `results` directory

## OCR Process

For each image:
1. Preprocessing:
   - Resize to standard height while maintaining aspect ratio
   - Extract region of interest (middle third of image)
   - Convert to grayscale
   - Apply adaptive thresholding
   - Denoise to remove speckles

2. OCR Processing:
   - Each engine processes the preprocessed image
   - Results are collected and timed
   - ID numbers are extracted using regex pattern

3. ID Number Extraction:
   - Regex pattern matches South African ID format:
     ```python
     r'\b\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4}\d{1}\b'
     ```
   - Pattern validates:
     - First 6 digits: YYMMDD (date of birth)
     - Next 4 digits: SSSS (sequence number)
     - Last digit: C (citizenship status)

## Performance Metrics

The evaluation report includes:
- Accuracy (exact matches with ground truth)
- Partial matches (when ID number is partially detected)
- No matches (failed detections)
- Average processing time per image

## Current Results

Based on recent evaluations:
- PaddleOCR: Best accuracy for complete matches (10%)
- EasyOCR: Higher rate of partial matches (20%)
- Tesseract: Fastest processing time but lower accuracy

## Notes

- The script processes images in parallel using ThreadPoolExecutor
- Results are saved in both detailed (JSON) and summary (Markdown) formats
- Error handling and logging are implemented throughout
- The script is configurable through various parameters in the code

## Future Improvements

- Fine-tune preprocessing parameters for better accuracy
- Implement additional OCR engines as they become available
- Add support for different ID document formats
- Improve ID number extraction logic
- Add configuration file for easy parameter tuning