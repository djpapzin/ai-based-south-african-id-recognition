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
import numpy as np
import argparse
import logging

# Configure logging
logging.getLogger("detectron2").setLevel(logging.WARNING)
logging.getLogger("fvcore").setLevel(logging.WARNING)

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize OCR engines
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

# Initialize document classifier
document_classifier = DocumentClassifier('models/classification_model_final.pth')

# Register dataset metadata
thing_classes = [
    'bottom_left_corner', 'bottom_right_corner', 'citizenship_status',
    'country_of_birth', 'date_of_birth', 'face', 'id_document', 'id_number',
    'names', 'nationality', 'sex', 'signature', 'surname', 'top_left_corner',
    'top_right_corner'
]

# Register metadata
if "sa_id" not in MetadataCatalog:
    MetadataCatalog.get("sa_id").set(thing_classes=thing_classes)

def clean_text(text, field_type):
    """Clean OCR text based on field type"""
    text = text.strip()
    if not text:
        return ""
        
    if field_type == 'id_number':
        # Keep only digits
        return ''.join(c for c in text if c.isdigit())
    elif field_type in ['date_of_birth']:
        # Keep the original date format, just clean extra whitespace
        return ' '.join(text.split())
    else:
        # Remove extra whitespace and newlines
        return ' '.join(text.split())

def process_segment_with_ocr(segment_img, label):
    """Process a segment with OCR and return results"""
    # Skip OCR for certain fields
    if label in ['face', 'id_document', 'signature']:
        return {'paddle_ocr': '', 'tesseract_ocr': ''}
    
    # Save original image dimensions for visualization
    h, w = segment_img.shape[:2]
    
    # Create a copy for preprocessing visualization
    preprocess_img = segment_img.copy()
    
    # Common preprocessing steps
    gray = cv2.cvtColor(segment_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # PaddleOCR
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        paddle_result = ocr.ocr(binary, cls=True)
        
        # Extract text and boxes
        paddle_text = ""
        paddle_boxes = []
        if paddle_result[0]:
            for line in paddle_result[0]:
                box, (text, conf) = line
                paddle_text += text + " "
                paddle_boxes.append(np.array(box).astype(np.int32))
        
        # Draw PaddleOCR boxes
        vis_img = preprocess_img.copy()
        cv2.putText(vis_img, "PaddleOCR Detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        for box in paddle_boxes:
            cv2.polylines(vis_img, [box], True, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"PaddleOCR error: {str(e)}")
        paddle_text = ""
        vis_img = preprocess_img.copy()
    
    # Tesseract
    try:
        tesseract_text = pytesseract.image_to_string(binary)
        
        # Get Tesseract bounding boxes
        tesseract_data = pytesseract.image_to_data(binary, output_type=pytesseract.Output.DICT)
        
        # Draw Tesseract boxes
        tesseract_img = preprocess_img.copy()
        cv2.putText(tesseract_img, "Tesseract Detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        for i, conf in enumerate(tesseract_data['conf']):
            if conf > 0:  # Filter out low confidence detections
                x = tesseract_data['left'][i]
                y = tesseract_data['top'][i]
                w = tesseract_data['width'][i]
                h = tesseract_data['height'][i]
                cv2.rectangle(tesseract_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    except Exception as e:
        print(f"Tesseract error: {str(e)}")
        tesseract_text = ""
        tesseract_img = preprocess_img.copy()
    
    return {
        'paddle_ocr': paddle_text.strip(),
        'tesseract_ocr': tesseract_text.strip(),
        'preprocess_image': preprocess_img,
        'paddle_vis': vis_img,
        'tesseract_vis': tesseract_img,
        'binary_image': binary
    }

def process_segment_with_ocr_segment_path(segment_path, field_type):
    """Process a segment with OCR based on field type"""
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    # Read the image
    image = cv2.imread(segment_path)
    if image is None:
        return {"error": f"Could not read image at {segment_path}"}
    
    # Run OCR
    try:
        result = ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return {"text": "", "confidence": 0.0}
        
        # Extract text and confidence
        text_results = []
        for line in result[0]:
            text = line[1][0]  # Get the recognized text
            confidence = float(line[1][1])  # Get the confidence score
            text_results.append({
                "text": text,
                "confidence": confidence
            })
        
        return text_results
        
    except Exception as e:
        return {"error": str(e)}

def run_inference(image_path, model_path, confidence_threshold=0.5):
    """Run inference on a single image"""
    results = {
        "classification": {},
        "segments": []
    }
    
    # First classify the document type
    try:
        doc_type, confidence = document_classifier.classify(image_path)
        print(f"\nDocument Classification:")
        print(f"Type: {doc_type}")
        print(f"Confidence: {confidence:.2%}")
        
        results["classification"] = {
            "type": doc_type,
            "confidence": float(confidence)
        }
    except Exception as e:
        print(f"Warning: Document classification failed: {str(e)}")
        doc_type = None
        confidence = 0.0
        results["classification"] = {
            "error": str(e)
        }

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
    
    # Create visualization
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("sa_id"), scale=1.0)
    visualization = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    visualization = visualization.get_image()[:, :, ::-1]
    
    # Add visualization and results to output
    outputs["visualization"] = visualization
    outputs["results"] = results
    
    return outputs

def save_segments(image_path, outputs, output_dir, config=None):
    """Save detected segments and perform OCR on them."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get predictions
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    # Get metadata
    metadata = MetadataCatalog.get("sa_id_train")
    class_names = metadata.thing_classes
    
    segments = []
    
    # Process each detected segment
    for box, score, class_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[class_id]
        
        # Create segment info
        segment_info = {
            'label': class_name,
            'confidence': float(score),
            'bbox': [int(coord) for coord in box],
            'class_id': int(class_id)
        }
        
        # Extract segment image
        segment_img = image[y1:y2, x1:x2]
        if segment_img.size == 0:
            continue
            
        # Save segment image
        segment_filename = f"{class_name}_{len(segments)}.jpg"
        segment_path = os.path.join(output_dir, segment_filename)
        cv2.imwrite(segment_path, segment_img)
        segment_info['image_path'] = segment_path
        
        # Save OCR results
        ocr_text_path = os.path.join(output_dir, f"{class_name}_{len(segments)}_ocr.txt")
        segment_info['ocr_text_path'] = ocr_text_path
        
        try:
            # Run OCR on segment
            paddle_result = ocr.ocr(segment_img, cls=True)
            tesseract_result = pytesseract.image_to_string(segment_img)
            
            # Save OCR results to text file
            with open(ocr_text_path, 'w', encoding='utf-8') as f:
                f.write("PaddleOCR Result:\n")
                if paddle_result and len(paddle_result) > 0 and len(paddle_result[0]) > 0:
                    for line in paddle_result[0]:
                        box, (text, conf) = line
                        f.write(f"Box: {box}\n")
                        f.write(f"Text: {text}\n")
                        f.write(f"Confidence: {conf}\n")
                        f.write(f"Raw: {text}\n")
                f.write("\nTesseract Result:\n")
                f.write(tesseract_result)
        
        except Exception as e:
            print(f"Error performing OCR on segment {segment_filename}: {str(e)}")
            continue
        
        segments.append(segment_info)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    metadata = {
        'original_image': image_path,
        'timestamp': datetime.now().isoformat(),
        'segments': segments
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run local inference on ID documents')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output', required=True, help='Path to output directory')
    args = parser.parse_args()

    # Set up paths
    IMAGE_DIR = args.input
    SAVE_DIR = args.output
    MODEL_PATH = "models/model_final.pth"
    CONFIDENCE_THRESHOLD = 0.5

    # Create output directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "segments"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "visualizations"), exist_ok=True)

    # Get list of images to process
    if os.path.isfile(IMAGE_DIR):
        image_files = [os.path.basename(IMAGE_DIR)]
        IMAGE_DIR = os.path.dirname(IMAGE_DIR)
    else:
        image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    for image_file in image_files:
        try:
            print(f"\nProcessing: {image_file}")
            image_path = os.path.join(IMAGE_DIR, image_file)
            
            # Run inference
            outputs = run_inference(image_path, MODEL_PATH, CONFIDENCE_THRESHOLD)
            
            # Save segments and results
            segments_dir = os.path.join(SAVE_DIR, "segments")
            classification_result = outputs.get("results", {}).get("classification", None)
            save_segments(image_path, outputs, segments_dir, classification_result)
            
            # Save visualization
            vis_dir = os.path.join(SAVE_DIR, "visualizations")
            vis_path = os.path.join(vis_dir, f"detected_{os.path.basename(image_path)}")
            cv2.imwrite(vis_path, outputs.get("visualization", None))
            print(f"\nSaved visualization to: {vis_path}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    print("\nProcessing complete!")
