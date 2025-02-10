# South African ID Document Recognition System - Setup and Usage Guide

## Environment Options

### 1. Google Colab (Training Environment - Completed)
- Training phase completed
- Model successfully trained and exported
- No further development in Colab environment
- All future development and testing to be done locally
- Colab notebook preserved for reference and future retraining if needed

Note: Development has transitioned to local Windows environment before final Linux server deployment.

### 2. Local Development Environment
The following instructions are for setting up the local inference environment.

#### System Requirements
- Windows operating system
- Anaconda or Miniconda installed
- At least 5GB of free disk space
- Internet connection for downloading dependencies
- (Optional) NVIDIA GPU with CUDA support

## Environment Setup

### 1. Create and Activate Conda Environment
```powershell
# Create new environment
conda create -n detectron2_env python=3.9
```

### 2. Activate the Environment
You have two options to activate and use the environment:

#### Option 1: Activate the environment (Recommended)
```powershell
conda activate detectron2_env
```
After activation, you can run scripts using regular python commands:
```powershell
python run_batch_inference.py
```

#### Option 2: Use full path to Python executable
If you're having issues with environment activation, you can use the full path:
```powershell
C:\Users\lfana\anaconda3\envs\detectron2_env\python.exe run_batch_inference.py
```

**IMPORTANT**: Always ensure you're using the correct environment before running any scripts. The scripts require specific dependencies that are only available in the `detectron2_env` environment.

### 3. Install Dependencies
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
pip install tqdm  # For progress bars
```

### 4. Additional Setup
1. Download and install Tesseract OCR:
   - Download the Windows installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to the default location: `C:\Program Files\Tesseract-OCR\`
   - Add the installation directory to your system PATH

### 5. Verify Installation
Run these commands to verify your setup:
```powershell
# Activate environment
conda activate detectron2_env

# Verify Python version
python --version  # Should show Python 3.9.x

# Verify key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import detectron2; print(f'Detectron2: {detectron2.__version__}')"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR: OK')"
```

## Running Scripts

### Batch Inference Script
The `run_batch_inference.py` script processes multiple ID images and generates structured output:

1. **Prepare Images**:
   - Place new ID images in: `test_dataset/new_ids/`
   - Place old ID images in: `test_dataset/old_ids/`

2. **Run the Script**:
   ```powershell
   # Option 1: With activated environment
   conda activate detectron2_env
   python run_batch_inference.py

   # Option 2: Using full path
   C:\Users\lfana\anaconda3\envs\detectron2_env\python.exe run_batch_inference.py
   ```

3. **Output**:
   - OCR results will be saved in: `ground_truth/ocr_results/`
   - Progress bar will show processing status
   - Any errors will be logged to the console

### Troubleshooting
If you encounter issues:
1. Verify you're using the correct environment
2. Check all dependencies are installed
3. Ensure model files are present in the `models` directory
4. Verify Tesseract OCR is properly installed and in PATH

## Project Structure
```
Machine Learning/
├── models/                    # Contains trained models
│   ├── classification_model_final.pth
│   └── model_final.pth       # Exported from Colab training
├── test_images/              # Directory for images to process
├── outputs/                  # Output directory
│   ├── classified/          # JSON results
│   └── text_results/        # Text format results
├── run_batch_inference.py    # Main inference script
└── document_classifier.py    # Document classification module
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
