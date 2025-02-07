from document_classifier import DocumentClassifier
import os
import json
from datetime import datetime
from paddleocr import PaddleOCR
import cv2
from local_inference import run_inference, save_segments

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
            
            # Save combined results
            output_dir = "outputs/classified"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_{timestamp}.json")
            
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"Results saved to: {result_file}")
            
        except Exception as e:
            print(f"\nError processing {image_file}: {str(e)}")
    
    print("\nProcessing complete. Results saved in outputs/classified/")

if __name__ == "__main__":
    main()
