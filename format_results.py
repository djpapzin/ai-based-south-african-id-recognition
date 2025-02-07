import json
import os
from pathlib import Path

def format_ocr_results(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create output text
    output = []
    output.append(f"File name: {data['image_path']}")
    output.append(f"Classification: {'New' if data['classification']['document_type'] == 'new_id' else 'Old'} ID")
    output.append(f"Confidence: {data['classification']['confidence']:.2f}")
    output.append("\nSegmented Files and OCR Results:")
    output.append("-" * 50)
    
    # Process each segment that has OCR results
    for segment in data['segments']['segments']:
        if 'paddle_ocr' in segment or 'tesseract_ocr' in segment:
            output.append(f"\nLabel: {segment['label']}")
            output.append(f"Segment File: {segment['segment_path']}")
            output.append(f"Confidence: {segment['confidence']:.2f}")
            if 'tesseract_ocr' in segment:
                output.append(f"Tesseract OCR: {segment['tesseract_ocr']}")
            if 'paddle_ocr' in segment:
                output.append(f"Paddle OCR: {segment['paddle_ocr']}")
            output.append("-" * 30)
    
    return "\n".join(output)

def process_all_json_files():
    # Directory containing the JSON files
    json_dir = Path("outputs/classified")
    output_dir = Path("outputs/text_results")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Process each JSON file
    for json_file in json_dir.glob("*.json"):
        # Only process the full result files (the ones with larger size)
        if json_file.stat().st_size > 1000:  # Skip the small classification-only files
            try:
                formatted_text = format_ocr_results(json_file)
                
                # Create output text file
                output_file = output_dir / f"{json_file.stem}.txt"
                with open(output_file, 'w') as f:
                    f.write(formatted_text)
                
                print(f"Processed: {json_file.name}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {str(e)}")

if __name__ == "__main__":
    process_all_json_files()
