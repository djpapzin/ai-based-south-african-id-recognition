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
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
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
            
            # Process with OCR and save visualization
            ocr_results = process_segment_with_ocr(segment, label)
            
            # Save preprocessed and visualization images
            if label not in ['face', 'id_document', 'signature']:
                preprocess_path = os.path.join(segments_dir, f"{base_filename}_preprocess.jpg")
                cv2.imwrite(preprocess_path, ocr_results['preprocess_image'])
                
                binary_path = os.path.join(segments_dir, f"{base_filename}_binary.jpg")
                cv2.imwrite(binary_path, ocr_results['binary_image'])
                
                paddle_vis_path = os.path.join(segments_dir, f"{base_filename}_paddle_boxes.jpg")
                cv2.imwrite(paddle_vis_path, ocr_results['paddle_vis'])
                
                tesseract_vis_path = os.path.join(segments_dir, f"{base_filename}_tesseract_boxes.jpg")
                cv2.imwrite(tesseract_vis_path, ocr_results['tesseract_vis'])
            
            # Create OCR results text file
            ocr_text_path = os.path.join(segments_dir, f"{base_filename}.txt")
            with open(ocr_text_path, 'w', encoding='utf-8') as f:
                f.write(f"Field: {label}\n")
                f.write(f"Confidence: {score:.2%}\n")
                if label not in ['face', 'id_document', 'signature']:
                    f.write("\nPaddleOCR Result:\n")
                    f.write(f"Raw: {ocr_results['paddle_ocr']}\n")
                    f.write("\nTesseract Result:\n")
                    f.write(f"Raw: {ocr_results['tesseract_ocr']}\n")
            
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
                }
            }
            
            if label not in ['face', 'id_document', 'signature']:
                segment_info.update({
                    "paddle_ocr": ocr_results['paddle_ocr'],
                    "tesseract_ocr": ocr_results['tesseract_ocr'],
                    "preprocess_path": preprocess_path,
                    "binary_path": binary_path,
                    "paddle_vis_path": paddle_vis_path,
                    "tesseract_vis_path": tesseract_vis_path
                })
            
            results["segments"]["segments"].append(segment_info)
            
        # Create HTML preview file
        html_path = os.path.join(segments_dir, "preview.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Segment Results</title>
                <style>
                    .segment-container { 
                        margin-bottom: 40px;
                        border: 1px solid #ccc;
                        padding: 20px;
                        border-radius: 5px;
                    }
                    .segment-header {
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 10px;
                    }
                    .segment-images {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        margin-bottom: 20px;
                    }
                    .image-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    .segment-image { 
                        max-width: 100%;
                        height: auto;
                        margin-bottom: 5px;
                    }
                    .segment-text {
                        font-family: monospace;
                        white-space: pre-wrap;
                        background: #f5f5f5;
                        padding: 10px;
                        border-radius: 5px;
                    }
                    h2 { color: #333; }
                    .confidence { color: #0066cc; }
                    .image-label {
                        font-size: 0.9em;
                        color: #666;
                        text-align: center;
                    }
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
                    <div class="segment-header">
                        <h2>{segment["label"]}</h2>
                        <p class="confidence">Confidence: {segment["confidence"]:.2%}</p>
                    </div>
                """)
                
                if segment["label"] not in ['face', 'id_document', 'signature']:
                    # Add all visualization images
                    f.write("""<div class="segment-images">""")
                    
                    # Original segment
                    f.write(f"""
                    <div class="image-container">
                        <img src="{rel_img_path}" class="segment-image">
                        <div class="image-label">Original Segment</div>
                    </div>
                    """)
                    
                    # Preprocessed image
                    rel_preprocess_path = os.path.relpath(segment["preprocess_path"], segments_dir)
                    f.write(f"""
                    <div class="image-container">
                        <img src="{rel_preprocess_path}" class="segment-image">
                        <div class="image-label">Preprocessed</div>
                    </div>
                    """)
                    
                    # Binary image
                    rel_binary_path = os.path.relpath(segment["binary_path"], segments_dir)
                    f.write(f"""
                    <div class="image-container">
                        <img src="{rel_binary_path}" class="segment-image">
                        <div class="image-label">Binary</div>
                    </div>
                    """)
                    
                    # PaddleOCR visualization
                    rel_paddle_path = os.path.relpath(segment["paddle_vis_path"], segments_dir)
                    f.write(f"""
                    <div class="image-container">
                        <img src="{rel_paddle_path}" class="segment-image">
                        <div class="image-label">PaddleOCR Detections</div>
                    </div>
                    """)
                    
                    # Tesseract visualization
                    rel_tesseract_path = os.path.relpath(segment["tesseract_vis_path"], segments_dir)
                    f.write(f"""
                    <div class="image-container">
                        <img src="{rel_tesseract_path}" class="segment-image">
                        <div class="image-label">Tesseract Detections</div>
                    </div>
                    """)
                    
                    f.write("</div>")  # Close segment-images div
                else:
                    # Just show the original segment for face/id_document/signature
                    f.write(f"""
                    <div class="image-container">
                        <img src="{rel_img_path}" class="segment-image">
                    </div>
                    """)
                
                f.write(f"""
                    <div class="segment-text">
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
