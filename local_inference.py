import os
import cv2
import torch
import json
import pytesseract
from datetime import datetime
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
from paddleocr import PaddleOCR
from document_classifier import DocumentClassifier

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize OCR engines
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize document classifier
document_classifier = DocumentClassifier('classification_model_final.pth')

# Register dataset metadata
thing_classes = [
    'bottom_left_corner', 'bottom_right_corner', 'citizenship_status',
    'country_of_birth', 'date_of_birth', 'face', 'id_document', 'id_number',
    'names', 'nationality', 'sex', 'signature', 'surname', 'top_left_corner',
    'top_right_corner'
]

# Register metadata
if "sa_id_val" not in MetadataCatalog:
    MetadataCatalog.get("sa_id_val").set(thing_classes=thing_classes)

def clean_text(text, field_type):
    """Clean OCR text based on field type"""
    text = text.strip()
    if not text:
        return ""
        
    if field_type == 'id_number':
        # Keep only digits
        return ''.join(c for c in text if c.isdigit())
    elif field_type in ['date_of_birth']:
        # Keep digits and common date separators
        return ''.join(c for c in text if c.isdigit() or c in '/-.')
    else:
        # Remove extra whitespace and newlines
        return ' '.join(text.split())

def process_segment_with_ocr(image_path, field_type):
    """Process a segment with both Pytesseract and PaddleOCR"""
    results = []
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return {'error': "Could not read image"}
    
    # Preprocess for OCR
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if field_type in ['id_number', 'date_of_birth']:
        processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    
    # Pytesseract OCR
    try:
        config = ''
        if field_type == 'id_number':
            config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        elif field_type == 'date_of_birth':
            config = '--psm 7 -c tessedit_char_whitelist=0123456789/-'
        else:
            config = '--psm 6'
            
        tesseract_text = pytesseract.image_to_string(processed, config=config)
        results.append({
            'engine': 'Pytesseract',
            'text': clean_text(tesseract_text, field_type)
        })
    except Exception as e:
        print(f"Pytesseract error: {str(e)}")
        results.append({
            'engine': 'Pytesseract',
            'text': ''
        })
    
    # PaddleOCR
    try:
        paddle_result = ocr.ocr(image, cls=True)
        paddle_text = ""
        if paddle_result[0]:
            texts = [line[1][0] for line in paddle_result[0]]
            paddle_text = " ".join(texts)
            paddle_text = clean_text(paddle_text, field_type)
        
        results.append({
            'engine': 'PaddleOCR',
            'text': paddle_text
        })
    except Exception as e:
        print(f"PaddleOCR error: {str(e)}")
        results.append({
            'engine': 'PaddleOCR',
            'text': ''
        })
    
    return results

def run_inference(image_path, model_path, confidence_threshold=0.5):
    """Run inference on a single image"""
    # First classify the document type
    try:
        doc_type, confidence = document_classifier.classify(image_path)
        print(f"\nDocument Classification:")
        print(f"Type: {doc_type}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Warning: Document classification failed: {str(e)}")
        doc_type = None
        confidence = 0.0

    # Configure model based on document type
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold

    # Initialize predictor
    predictor = DefaultPredictor(cfg)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Run inference
    outputs = predictor(image)
    
    # Add document type to outputs for downstream processing
    outputs["document_type"] = {
        "class": doc_type,
        "confidence": confidence
    }
    
    # Visualize results
    v = Visualizer(image[:, :, ::-1],
                  metadata=MetadataCatalog.get("sa_id_val"),
                  scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    return outputs, out.get_image()[:, :, ::-1]

def save_segments(image_path, outputs, save_dir):
    """Save detected segments with their labels and OCR results"""
    # Create directory for this image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_dir = os.path.join(save_dir, base_name)
    os.makedirs(image_dir, exist_ok=True)
    
    # Save document classification result
    doc_type = outputs.get("document_type", {"class": "unknown", "confidence": 0.0})
    
    # Process detections
    instances = outputs["instances"].to("cpu")
    metadata = MetadataCatalog.get("sa_id_val")
    field_counts = {}
    detection_results = {}
    
    for i in range(len(instances)):
        box = instances.pred_boxes[i].tensor[0].numpy().astype(int)
        label = instances.pred_classes[i].item()
        score = instances.scores[i].item()
        class_name = metadata.thing_classes[label]
        
        # Handle duplicate fields
        if class_name in field_counts:
            field_counts[class_name] += 1
            filename = f"{class_name}_{field_counts[class_name]}"
        else:
            field_counts[class_name] = 1
            filename = class_name
        
        # Extract and save segment
        x1, y1, x2, y2 = box
        padding = 5
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(instances.image_size[1], x2 + padding), min(instances.image_size[0], y2 + padding)
        segment = instances.image_tensor[i, :, y1:y2, x1:x2].numpy().astype(np.uint8)
        
        # Save image segment
        image_filename = f"{filename}.jpg"
        cv2.imwrite(os.path.join(image_dir, image_filename), segment)
        
        # Process with OCR and save results
        ocr_results = process_segment_with_ocr(os.path.join(image_dir, image_filename), class_name)
        
        # Save OCR results to text file
        text_filename = f"{filename}.txt"
        with open(os.path.join(image_dir, text_filename), 'w', encoding='utf-8') as f:
            for result in ocr_results:
                f.write(f"{result['engine']} OCR: {result['text']}\n")
        
        # Update results dictionary
        detection_results[filename] = {
            "class": class_name,
            "confidence": float(score),
            "bbox": [int(x) for x in box],
            "ocr_results": ocr_results
        }
    
    # Update metadata with document classification
    metadata_dict = {
        "timestamp": datetime.now().isoformat(),
        "document_type": {
            "class": doc_type["class"],
            "confidence": doc_type["confidence"]
        },
        "detections": detection_results
    }
    
    with open(os.path.join(image_dir, "detection_metadata.json"), 'w') as f:
        json.dump(metadata_dict, f, indent=2)

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/model_final.pth"  # Path to your model weights
    IMAGE_DIR = "test_images"  # Directory containing test images
    SAVE_DIR = "detected_segments"  # Directory to save cropped segments
    CONFIDENCE_THRESHOLD = 0.5
    
    # Verify model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure model_final.pth is in the same directory as this script")
        exit(1)
    
    # Create output directories
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Process all images in the directory
    image_files = [f for f in os.listdir(IMAGE_DIR) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        print("Please add some images to the test_images directory")
        exit(1)
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        try:
            print(f"\nProcessing: {image_file}")
            image_path = os.path.join(IMAGE_DIR, image_file)
            outputs, visualization = run_inference(image_path, MODEL_PATH, CONFIDENCE_THRESHOLD)
            save_segments(image_path, outputs, SAVE_DIR)
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, visualization)
            print(f"\nSaved visualization to: {output_path}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    print("\nProcessing complete!")
