import torch
from document_classifier import DocumentClassifier
import os

def test_classifier():
    """Test the document classifier on sample images"""
    # Use the trained model from local_inference.py
    model_path = "models/model_final.pth"
    classifier = DocumentClassifier(model_path)
    
    # Get list of test images
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"\nCreated {test_dir} directory. Please add some test images.")
        return
        
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"\nNo test images found in {test_dir}. Please add some .jpg, .jpeg, or .png files.")
        return
    
    print("\nTesting document classifier...")
    print("-" * 50)
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        try:
            # Run classification
            doc_type, confidence = classifier.classify(image_path)
            
            # Print results
            print(f"\nImage: {image_file}")
            print(f"Predicted Type: {doc_type}")
            print(f"Confidence: {confidence:.2%}")
            
        except Exception as e:
            print(f"\nError processing {image_file}: {str(e)}")
            continue

if __name__ == "__main__":
    test_classifier()
