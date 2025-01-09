# ID Document Annotation Project

This project uses the Segment Anything Model (SAM) to assist in annotating ID documents. It provides two approaches:
1. A standalone Python script for automatic annotation
2. Integration with Label Studio for interactive annotation

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install segment-anything opencv-python numpy requests
```

3. Download SAM model weights:
```bash
python download_sam.py
```

## Project Structure

```
.
├── IDcopies/               # Directory containing ID document images
├── results/                # Output directory for annotations
├── sam_backend/           # Label Studio ML backend
│   ├── model.py          # SAM integration code
│   ├── requirements.txt  # Backend dependencies
│   └── Dockerfile       # Docker configuration for backend
├── id_annotator.py       # Standalone annotation script
├── download_sam.py       # Script to download SAM weights
└── README.md            # This file
```

## Usage

### Standalone Script
To process images using the standalone script:
```bash
python id_annotator.py
```
This will:
1. Process all JPEG images in the IDcopies directory
2. Generate annotations for each image
3. Save results in `results/annotations.json`

### Label Studio Integration

1. Start Label Studio:
```bash
label-studio start
```

2. Create a new project in Label Studio:
   - Choose "Object Detection with Bounding Boxes" template
   - Configure labels:
     - id_document
     - name
     - surname
     - date_of_birth
     - identity_number
     - face
     - stamp
     - signature

3. Build and run the SAM backend:
```bash
cd sam_backend
docker build -t sam-backend .
docker run -p 9090:8080 sam-backend
```

4. Connect the ML backend in Label Studio:
   - Go to Settings > Machine Learning
   - Add Model
   - URL: http://localhost:9090
   - Click "Validate and Save"

## Output Format

The annotations are saved in JSON format with the following structure:
```json
{
  "id_number": {
    "image_name": [
      {
        "bbox": [x, y, width, height],
        "score": confidence_score,
        "point": [x, y]
      }
    ]
  }
}
```

## Notes
- PDF files are currently skipped in the standalone script
- The SAM model runs on CPU by default for compatibility
- Adjust the points of interest in `id_annotator.py` to focus on specific regions
- Make sure to have sufficient disk space for the SAM model weights (~2.4GB)