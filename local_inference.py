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

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize OCR engines
ocr = PaddleOCR(use_angle_cls=True, lang='en')

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
        # Keep digits and common date separators
        return ''.join(c for c in text if c.isdigit() or c in '/-.')
    else:
        # Remove extra whitespace and newlines
        return ' '.join(text.split())

def process_segment_with_ocr(image, label):
    """Process a segment with OCR and return results from both engines"""
    try:
        results = {
            'paddle_ocr': '',
            'tesseract_ocr': ''
        }
        
        # Skip OCR for these labels
        if label in ['face', 'signature', 'id_document']:
            return results
            
        # PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        paddle_result = ocr.ocr(image, cls=True)
        
        if paddle_result and paddle_result[0]:
            texts = []
            for line in paddle_result[0]:
                if line[1][0]:  # Check if there's text detected
                    texts.append(line[1][0])  # Get the text content
            results['paddle_ocr'] = " ".join(texts)
            
        # Tesseract OCR
        # Convert image to RGB if it's not
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        results['tesseract_ocr'] = pytesseract.image_to_string(image).strip()
        
        return results
    except Exception as e:
        print(f"OCR Error for {label}: {str(e)}")
        return {'paddle_ocr': '', 'tesseract_ocr': ''}

def process_segment_with_ocr_segment_path(segment_path, field_type):
    """Process a segment with OCR based on field type"""
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
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

def save_segments(image_path, outputs, save_dir, classification_result=None):
    """Save detected segments and process with OCR"""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Get predictions
        predictions = outputs["instances"].to("cpu")
        boxes = predictions.pred_boxes.tensor.numpy()
        classes = predictions.pred_classes.numpy()
        scores = predictions.scores.numpy()
        
        # Create directory for this image's segments
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        segments_dir = os.path.join(save_dir, image_name)
        os.makedirs(segments_dir, exist_ok=True)
        
        # Save detection and classification results
        results = outputs.get("results", {})
        results["image_path"] = image_path
        results["segments"] = {
            "directory": segments_dir,
            "segments": []
        }
        
        # Process each detected segment
        for i, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
            # Get field label
            label = thing_classes[class_id]
            
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Crop segment from image
            segment = image[y1:y2, x1:x2]
            
            # Generate base filename without extension
            base_filename = f"{label}_{i+1}"
            
            # Save segment image
            segment_path = os.path.join(segments_dir, f"{base_filename}.jpg")
            cv2.imwrite(segment_path, segment)
            
            # Process with OCR
            ocr_results = process_segment_with_ocr(segment, label)
            
            # Clean OCR results
            cleaned_paddle = clean_text(ocr_results['paddle_ocr'], label)
            cleaned_tesseract = clean_text(ocr_results['tesseract_ocr'], label)
            
            # Create OCR results text file
            ocr_text_path = os.path.join(segments_dir, f"{base_filename}.txt")
            with open(ocr_text_path, 'w', encoding='utf-8') as f:
                f.write(f"Field: {label}\n")
                f.write(f"Confidence: {score:.2%}\n")
                f.write("\nPaddleOCR Result:\n")
                f.write(f"Raw: {ocr_results['paddle_ocr']}\n")
                f.write(f"Cleaned: {cleaned_paddle}\n")
                f.write("\nTesseract Result:\n")
                f.write(f"Raw: {ocr_results['tesseract_ocr']}\n")
                f.write(f"Cleaned: {cleaned_tesseract}\n")
            
            # Add to results
            segment_info = {
                "label": label,
                "confidence": float(score),
                "segment_path": segment_path,
                "ocr_text_path": ocr_text_path,
                "coordinates": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "paddle_ocr": cleaned_paddle,
                "tesseract_ocr": cleaned_tesseract
            }
            results["segments"]["segments"].append(segment_info)
            
            # Create HTML preview file for easy viewing
            html_path = os.path.join(segments_dir, "preview.html")
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Segment Results</title>
                    <style>
                        .segment-container { 
                            display: flex; 
                            margin-bottom: 20px;
                            border: 1px solid #ccc;
                            padding: 10px;
                            border-radius: 5px;
                        }
                        .segment-image { 
                            max-width: 300px; 
                            margin-right: 20px;
                        }
                        .segment-text {
                            font-family: monospace;
                            white-space: pre-wrap;
                        }
                        h2 { color: #333; }
                        .confidence { color: #0066cc; }
                    </style>
                </head>
                <body>
                    <h1>ID Document Segments and OCR Results</h1>
                """)
                
                # Add each segment to the HTML
                for segment in results["segments"]["segments"]:
                    rel_img_path = os.path.relpath(segment["segment_path"], segments_dir)
                    with open(segment["ocr_text_path"], 'r', encoding='utf-8') as txt:
                        ocr_content = txt.read()
                    
                    f.write(f"""
                    <div class="segment-container">
                        <div>
                            <h2>{segment["label"]}</h2>
                            <img src="{rel_img_path}" class="segment-image">
                        </div>
                        <div class="segment-text">
                            <p class="confidence">Confidence: {segment["confidence"]:.2%}</p>
                            <pre>{ocr_content}</pre>
                        </div>
                    </div>
                    """)
                
                f.write("""
                </body>
                </html>
                """)
        
        return results
        
    except Exception as e:
        print(f"Error saving segments: {str(e)}")
        return {"error": str(e)}

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
