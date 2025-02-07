# South African ID Document Recognition System - Setup and Usage Guide

## System Requirements
- Windows operating system
- Anaconda or Miniconda installed
- At least 5GB of free disk space
- Internet connection for downloading dependencies

## Environment Setup

### 1. Create and Activate Conda Environment
```powershell
# Create new environment
conda create -n detectron2_env python=3.9
conda activate detectron2_env
```

### 2. Install Dependencies
Install the required packages in the following order:

```powershell
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
pip install paddleocr
pip install pytesseract
pip install opencv-python
```

### 3. Additional Setup
1. Download and install Tesseract OCR:
   - Download the Windows installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to the default location (Usually C:\Program Files\Tesseract-OCR)
   - Add the installation directory to your system PATH

## Project Structure
```
Machine Learning/
├── models/                    # Contains trained models
│   ├── classification_model_final.pth
│   └── model_final.pth
├── test_images/              # Directory for images to process
├── outputs/                  # Output directory
│   ├── classified/          # JSON results
│   └── text_results/        # Text format results
├── run_batch_inference.py    # Main inference script
└── document_classifier.py    # Document classification module
```

## Running the Script

### 1. Prepare Input Images
- Place the ID document images you want to process in the `test_images` directory
- Supported formats: JPG, PNG
- Images should be clear and well-lit for best results

### 2. Run the Script
```powershell
# Activate the environment (if not already activated)
conda activate detectron2_env

# Run the inference script
python run_batch_inference.py
```

## Output Format

### JSON Output (in outputs/classified/)
```json
{
    "file_name": "example.jpg",
    "document_type": "new_id",
    "confidence": 89.84,
    "segments": [
        {
            "label": "face",
            "confidence": 99.61,
            "bbox": [x1, y1, x2, y2]
        },
        {
            "label": "names",
            "confidence": 98.99,
            "text": {
                "paddle_ocr": "EXAMPLE NAME",
                "tesseract": "EXAMPLE NAME"
            }
        }
        // ... other segments
    ]
}
```

### Text Output (in outputs/text_results/)
```
Image: example.jpg
Classified as: new_id
Confidence: 89.84%

Document Classification:
Type: new_id
Confidence: 84.21%

Detected Segments:
--------------------------------------------------
Label: face
Confidence: 99.61%
--------------------------------------------------
Label: names
Confidence: 98.99%
PaddleOCR: EXAMPLE NAME
Tesseract: EXAMPLE NAME
--------------------------------------------------
// ... other segments
```

## Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError: No module named 'torch'**
   - Solution: Ensure you're in the correct conda environment
   - Run: `conda activate detectron2_env`
   - Reinstall PyTorch if needed

2. **ModuleNotFoundError: No module named 'detectron2'**
   - Solution: Reinstall detectron2
   - Run: `pip install 'git+https://github.com/facebookresearch/detectron2.git'`

3. **TesseractNotFound Error**
   - Solution: Ensure Tesseract is installed and in your PATH
   - Check if you can run `tesseract --version` in PowerShell

4. **CUDA/GPU Related Errors**
   - Solution: The script will run on CPU by default
   - For GPU support, ensure correct CUDA version is installed

### Performance Tips

1. Image Quality:
   - Use clear, well-lit images
   - Ensure the ID document is properly aligned
   - Avoid glare and shadows

2. Processing Speed:
   - CPU processing is slower but more reliable
   - Batch processing multiple images will take longer
   - Keep test_images directory organized

## Support

For additional support or to report issues:
1. Check the error messages in the console output
2. Verify all dependencies are correctly installed
3. Ensure you're using the correct Python and conda environment
4. Check if all required model files are present in the models directory

## Version Information
- Python: 3.9
- PyTorch: Latest stable with CUDA 11.8
- Detectron2: Latest from GitHub
- PaddleOCR: Latest stable
- Tesseract: 5.0.0 or later
