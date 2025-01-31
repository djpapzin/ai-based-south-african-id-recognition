"""Configuration settings for the ID Processing Pipeline."""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Model paths
CLASSIFICATION_MODEL_PATH = os.path.join(MODELS_DIR, "classification_model.pth")
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "detection_model.pth")

# OCR Configuration
TESSERACT_CONFIG = "--oem 3 --psm 6"

# Field names and types
FIELD_TYPES = {
    "id_document": "box",
    "corners": "points",
    "face": "image",
    "signature": "image",
    "id_number": "text",
    "names": "text",
    "surname": "text",
    "date_of_birth": "text",
    "country_of_birth": "text",
    "sex": "text",
    "citizenship": "text",
    "nationality": "text"
}

# Output configuration
JSON_OUTPUT_TEMPLATE = {
    "document_type": None,  # "old_id" or "new_id"
    "confidence_scores": {
        "classification": None,
        "detection": {},
        "ocr": {}
    },
    "extracted_fields": {},
    "extracted_images": {
        "face": None,
        "signature": None
    },
    "processing_time": None
}
