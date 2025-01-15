import os
import cv2
import json
import torch
import logging
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from ocr_evaluator import OCREvaluator
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detect_and_ocr.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IDCardProcessor:
    def __init__(self, model_path, confidence_threshold=0.7):
        """Initialize the ID card processor with model and OCR."""
        self.setup_detector(model_path, confidence_threshold)
        self.ocr_evaluator = OCREvaluator()
        
    def setup_detector(self, model_path, confidence_threshold):
        """Set up the Detectron2 model for inference."""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Update based on your number of classes
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = "cpu"  # Ensure CPU usage
        self.predictor = DefaultPredictor(cfg)
        logger.info("Detector model loaded successfully")

    def process_image(self, image_path, output_dir):
        """Process a single image through detection and OCR."""
        # Create output directories
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, image_name)
        regions_dir = os.path.join(image_output_dir, "regions")
        os.makedirs(regions_dir, exist_ok=True)

        # Read and process image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return None

        # Run detection
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()

        # Process each detection
        results = []
        class_names = ["id_number", "surname", "names", "nationality", 
                      "country_of_birth", "status", "sex"]  # Update with your classes

        for box, score, class_idx in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_idx]
            
            # Extract region
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                continue

            # Save region
            region_filename = f"{class_name}_{score:.2f}.jpg"
            region_path = os.path.join(regions_dir, region_filename)
            cv2.imwrite(region_path, region)

            # Perform OCR on region
            preprocessed_region = self.ocr_evaluator.preprocess_image(region)
            ocr_text = self.ocr_evaluator.perform_tesseract_ocr(preprocessed_region)

            result = {
                "class": class_name,
                "confidence": float(score),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "ocr_text": ocr_text,
                "region_path": region_path
            }
            results.append(result)

        # Save results
        results_file = os.path.join(image_output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump({
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "detections": results
            }, f, indent=2)

        # Save visualization
        self.save_visualization(image, results, image_output_dir)
        return results

    def save_visualization(self, image, results, output_dir):
        """Save visualization of detections and OCR results."""
        vis_image = image.copy()
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            class_name = result["class"]
            score = result["confidence"]
            ocr_text = result["ocr_text"]

            # Draw box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {ocr_text}"
            cv2.putText(vis_image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save visualization
        cv2.imwrite(os.path.join(output_dir, "visualization.jpg"), vis_image)

def main():
    parser = argparse.ArgumentParser(description="Process ID cards with detection and OCR")
    parser.add_argument("--model-path", required=True, help="Path to the trained model weights")
    parser.add_argument("--input", required=True, help="Path to input image or directory")
    parser.add_argument("--output", default="results", help="Path to output directory")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize processor
    processor = IDCardProcessor(args.model_path, args.confidence)

    # Process input
    if os.path.isfile(args.input):
        # Single image
        processor.process_image(args.input, args.output)
    else:
        # Directory of images
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input, filename)
                processor.process_image(image_path, args.output)

if __name__ == "__main__":
    main()
