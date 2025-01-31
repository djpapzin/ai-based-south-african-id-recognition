"""Main pipeline for ID document processing."""

import os
import time
import json
import cv2
import numpy as np
import pytesseract
from config import *

class IDProcessingPipeline:
    def __init__(self):
        self.setup_directories()
        # Models will be loaded here once training is complete
        self.classifier = None
        self.detector = None
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "json"), exist_ok=True)

    def load_models(self):
        """Load classification and detection models."""
        # TODO: Implement model loading once training is complete
        pass

    def process_image(self, image_path):
        """Process a single image through the entire pipeline."""
        start_time = time.time()
        results = self._initialize_results()

        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")

            # Run classification
            doc_type, class_confidence = self.classify_document(image)
            results["document_type"] = doc_type
            results["confidence_scores"]["classification"] = class_confidence

            # Detect corners and normalize
            corners = self.detect_corners(image)
            normalized_image = self.normalize_image(image, corners)

            # Detect fields
            fields = self.detect_fields(normalized_image)
            
            # Process each field
            for field_name, field_data in fields.items():
                if FIELD_TYPES[field_name] == "text":
                    text, confidence = self.extract_text(normalized_image, field_data)
                    results["extracted_fields"][field_name] = text
                    results["confidence_scores"]["ocr"][field_name] = confidence
                elif FIELD_TYPES[field_name] == "image":
                    image_path = self.save_field_image(normalized_image, field_data, field_name)
                    results["extracted_images"][field_name] = image_path

            results["processing_time"] = time.time() - start_time
            return results

        except Exception as e:
            results["error"] = str(e)
            return results

    def _initialize_results(self):
        """Initialize results dictionary."""
        return json.loads(json.dumps(JSON_OUTPUT_TEMPLATE))

    def classify_document(self, image):
        """Classify document as old_id or new_id."""
        # TODO: Implement classification
        pass

    def detect_corners(self, image):
        """Detect document corners for normalization."""
        # TODO: Implement corner detection
        pass

    def normalize_image(self, image, corners):
        """Normalize image perspective based on corners."""
        # TODO: Implement image normalization
        pass

    def detect_fields(self, image):
        """Detect and locate all fields in the image."""
        # TODO: Implement field detection using Detectron2
        pass

    def extract_text(self, image, field_data):
        """Extract text from a field using OCR."""
        x, y, w, h = field_data["bbox"]
        field_image = image[y:y+h, x:x+w]
        
        # OCR with confidence
        text = pytesseract.image_to_string(field_image, config=TESSERACT_CONFIG)
        confidence = pytesseract.image_to_data(field_image, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence for the field
        confidences = [float(conf) for conf in confidence["conf"] if conf != -1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return text.strip(), avg_confidence

    def save_field_image(self, image, field_data, field_name):
        """Save extracted field image (face/signature) to file."""
        x, y, w, h = field_data["bbox"]
        field_image = image[y:y+h, x:x+w]
        
        output_path = os.path.join(OUTPUT_DIR, "images", f"{field_name}.jpg")
        cv2.imwrite(output_path, field_image)
        
        return output_path

    def save_results(self, results, base_filename):
        """Save processing results to JSON file."""
        output_path = os.path.join(OUTPUT_DIR, "json", f"{base_filename}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return output_path
