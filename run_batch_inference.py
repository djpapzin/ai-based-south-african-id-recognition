from document_classifier import DocumentClassifier
import os
import json
from datetime import datetime
from paddleocr import PaddleOCR
import cv2
from local_inference import run_inference, save_segments
import re

def extract_date(text):
    """Extract and format date from OCR text."""
    # Try to match various date formats
    patterns = [
        r'(\d{2})\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*(\d{4})',  # DD MMM YYYY
        r'(\d{4})[/-](\d{2})[/-](\d{2})',  # YYYY-MM-DD or YYYY/MM/DD
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 3:
                if match.group(2).isdigit():  # YYYY-MM-DD format
                    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                else:  # DD MMM YYYY format
                    return f"{match.group(3)}-{match.group(2).upper()}-{match.group(1)}"
    return None

def extract_id_number(text, id_type):
    """Extract and format ID number based on ID type."""
    # Remove all spaces and non-digit characters
    digits = ''.join(filter(str.isdigit, text))
    
    if len(digits) == 13:
        if id_type == 'old':
            # Format as old ID: XXXXXX XXXX XX X
            return f"{digits[:6]} {digits[6:10]} {digits[10:12]} {digits[12]}"
        else:
            # New ID format: continuous 13 digits
            return digits
    return None

def extract_citizenship(text, id_type):
    """Extract and format citizenship status."""
    text = text.upper()
    if id_type == 'old':
        if 'S.A.BURGER' in text or 'S.A.CITIZEN' in text:
            return 'S.A.BURGER/S.A.CITIZEN'
        return 'S.A.CITIZEN'
    else:
        return 'CITIZEN'

def process_ocr_results(segments, id_type):
    """Process OCR results into structured format."""
    result = {
        'id_type': id_type,
        'id_number': None,
        'surname': None,
        'names': None,
        'nationality': 'RSA' if id_type == 'new' else None,
        'country_of_birth': None,
        'date_of_birth': None,
        'sex': None,
        'citizenship_status': None,
        'has_signature': id_type == 'new',
        'confidence': {
            'image_quality': 'clear',
            'uncertain_fields': []
        }
    }
    
    uncertain_fields = []
    
    for segment in segments['segments']:
        label = segment['label'].lower()
        ocr_text = segment.get('paddle_ocr', '') or segment.get('tesseract_ocr', '')
        
        if not ocr_text:
            continue
            
        confidence = segment.get('confidence', 0)
        if confidence < 0.7:  # Low confidence threshold
            uncertain_fields.append(label)
        
        if 'id_number' in label:
            result['id_number'] = extract_id_number(ocr_text, id_type)
        elif 'surname' in label:
            result['surname'] = ocr_text.strip().upper()
        elif 'names' in label or 'forename' in label:
            result['names'] = ocr_text.strip().upper()
        elif 'birth' in label and 'country' not in label:
            result['date_of_birth'] = extract_date(ocr_text)
        elif 'country' in label and 'birth' in label:
            result['country_of_birth'] = ocr_text.strip().upper()
        elif 'citizen' in label:
            result['citizenship_status'] = extract_citizenship(ocr_text, id_type)
        elif 'sex' in label:
            result['sex'] = ocr_text.strip().upper()
    
    # Update confidence
    if uncertain_fields:
        result['confidence']['uncertain_fields'] = uncertain_fields
        if len(uncertain_fields) > 2:
            result['confidence']['image_quality'] = 'poor'
        elif len(uncertain_fields) > 0:
            result['confidence']['image_quality'] = 'partial'
    
    return result

def format_results(results):
    """Format the results into a readable text format."""
    output = []
    output.append(f"File name: {results['filename']}")
    output.append(f"ID Type: {results['id_type']}")
    output.append(f"ID Number: {results['id_number']}")
    output.append(f"Surname: {results['surname']}")
    output.append(f"Names: {results['names']}")
    output.append(f"Nationality: {results['nationality']}")
    output.append(f"Country of Birth: {results['country_of_birth']}")
    output.append(f"Date of Birth: {results['date_of_birth']}")
    output.append(f"Sex: {results['sex']}")
    output.append(f"Citizenship Status: {results['citizenship_status']}")
    output.append(f"Has Signature: {results['has_signature']}")
    output.append(f"Confidence: {results['confidence']['image_quality']}")
    output.append(f"Uncertain Fields: {', '.join(results['confidence']['uncertain_fields'])}")
    
    return "\n".join(output)

def main():
    # Initialize the classifier and OCR
    model_path = "models/model_final.pth"
    classifier = DocumentClassifier(model_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Process both new and old IDs
    for id_type in ['new', 'old']:
        # Input directory for ground truth images
        image_dir = os.path.join('test_dataset', f'{id_type}_ids')
        # Output directory for OCR results
        output_dir = os.path.join('ground_truth', 'ocr_results', f'{id_type}_ids')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        results = []
        
        print(f"\nProcessing {len(image_files)} {id_type} ID images from {image_dir}...")
        print("-" * 50)
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            try:
                # Run Detectron2 inference and save segments
                outputs = run_inference(image_path, model_path)
                segments_dir = os.path.join("outputs", "segments", os.path.splitext(image_file)[0])
                segment_results = save_segments(image_path, outputs, segments_dir, None)
                
                # Process OCR results
                result = process_ocr_results(segment_results, id_type)
                result['filename'] = image_file
                results.append(result)
                
                print(f"Processed: {image_file}")
                
            except Exception as e:
                print(f"\nError processing {image_file}: {str(e)}")
        
        # Save results
        output_file = os.path.join(output_dir, 'results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create and save formatted text output
        text_output_dir = os.path.join(output_dir, 'text_results')
        os.makedirs(text_output_dir, exist_ok=True)
        for result in results:
            text_result_file = os.path.join(text_output_dir, f"{os.path.splitext(result['filename'])[0]}.txt")
            formatted_text = format_results(result)
            
            with open(text_result_file, 'w') as f:
                f.write(formatted_text)
        
        print(f"\nSaved {len(results)} results to {output_file}")
    
    print("\nProcessing complete. Results saved in ground_truth/ocr_results/")

if __name__ == "__main__":
    main()
