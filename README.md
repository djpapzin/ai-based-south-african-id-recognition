# South African ID Card OCR Evaluator

A Python tool for evaluating different OCR engines' performance on South African ID cards. The tool processes both image and PDF files, handles rotated documents, and generates detailed performance reports.

## Features

- Supports multiple OCR engines:
  - Tesseract OCR
  - EasyOCR
  - PaddleOCR
- Handles both images (.jpg, .jpeg, .png) and PDFs
- Automatic rotation correction for portrait-oriented images
- Region of Interest (ROI) extraction
- Parallel processing for improved performance
- Comprehensive preprocessing pipeline
- Detailed evaluation reports in both JSON and Markdown formats

## Prerequisites

### Required Software
1. Python 3.7+
2. Tesseract OCR
3. Poppler (for PDF processing)

### Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Download OCR models:
```bash
python download_ocr_models.py
```

This will download the required models for EasyOCR and PaddleOCR into the `ocr_models` directory.

## Directory Structure

Your ID documents should be organized in folders named with the ID number:
```
IDcopies/
├── 0001011726088/
│   ├── 0001011726088_F.jpg  # Front of ID
│   ├── 0001011726088_B.jpg  # Back of ID
│   └── 0001011726088_A.pdf  # Scanned PDF (optional)
├── 9001016001086/
│   ├── 9001016001086_F.jpg
│   └── 9001016001086_B.jpg
...
```

## Usage

### Basic Usage
```bash
python ocr_evaluator.py "path/to/IDcopies" --results-dir results
```

### Test Mode (Process only 10 folders)
```bash
python ocr_evaluator.py "path/to/IDcopies" --results-dir results --test-mode
```

## Output

The script generates:

1. Preprocessed Images:
   - `results/preprocessed_images/{id_number}/{image_name}_preprocessed.png` - Full image with ROI marked
   - `results/preprocessed_images/{id_number}/{image_name}_roi.png` - Extracted ROI

2. Reports:
   - `results/ocr_results.json` - Detailed JSON results
   - `results/ocr_evaluation_report.md` - Summary report

## Troubleshooting

### Common Issues

1. "Unable to get page count. Is poppler installed and in PATH?"
   - Solution: Install poppler and add it to your PATH
   - Windows users: Restart your terminal after adding to PATH

2. "Tesseract not available"
   - Solution: Install Tesseract OCR and verify the path in the script

3. Slow Processing
   - Use `--test-mode` for initial testing
   - Reduce `max_workers` if memory usage is high
   - Consider using GPU versions of OCR engines for better performance

## Current Limitations

- CPU-only implementation (GPU support available through OCR engines)
- Requires specific directory structure
- PDF processing requires Poppler installation
- Processing speed depends on hardware capabilities

## License

[Your License Here]