from document_classifier import DocumentClassifier
import os
import json
from datetime import datetime
from paddleocr import PaddleOCR
import cv2
from local_inference import run_inference, save_segments

def format_results(results):
    """Format the results into a readable text format."""
    output = []
    output.append(f"File name: {results['image_path']}")
    output.append(f"Classification: {'New' if results['classification']['document_type'] == 'new_id' else 'Old'} ID")
    output.append(f"Confidence: {results['classification']['confidence']:.2f}")
    output.append("\nSegmented Files and OCR Results:")
    output.append("-" * 50)
    
    # Process each segment that has OCR results
    for segment in results['segments']['segments']:
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

def main():
    # Initialize the classifier and OCR
    model_path = "models/model_final.pth"
    classifier = DocumentClassifier(model_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    # Get all images from test_images directory
    test_dir = "test_images"
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nProcessing {len(image_files)} images from {test_dir}...")
    print("-" * 50)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        try:
            # 1. Document Classification
            doc_type, confidence = classifier.classify(image_path)
            print(f"\nImage: {image_file}")
            print(f"Classified as: {doc_type}")
            print(f"Confidence: {confidence:.2%}")
            
            # Create classification result dictionary
            classification_result = {
                "document_type": doc_type,
                "confidence": confidence
            }
            
            # 2. Run Detectron2 inference and save segments
            outputs = run_inference(image_path, model_path)
            segments_dir = os.path.join("outputs", "segments", os.path.splitext(image_file)[0])
            
            # Add classification results to outputs
            if "results" not in outputs:
                outputs["results"] = {}
            outputs["results"]["classification"] = classification_result
            
            # Save segments with classification results
            segment_results = save_segments(image_path, outputs, segments_dir, classification_result)
            
            # Combine results
            results = {
                "image_path": image_path,
                "classification": classification_result,
                "segments": segment_results,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            # Save combined results as JSON
            output_dir = "outputs/classified"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Create and save formatted text output
            text_output_dir = "outputs/text_results"
            os.makedirs(text_output_dir, exist_ok=True)
            text_result_file = os.path.join(text_output_dir, f"{os.path.splitext(image_file)[0]}_{timestamp}.txt")
            
            formatted_text = format_results(results)
            with open(text_result_file, 'w') as f:
                f.write(formatted_text)
                
            print(f"Results saved to:")
            print(f"- JSON: {result_file}")
            print(f"- Text: {text_result_file}")
            
        except Exception as e:
            print(f"\nError processing {image_file}: {str(e)}")
    
    print("\nProcessing complete. Results saved in outputs/classified/ and outputs/text_results/")

if __name__ == "__main__":
    main()
