import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.engine import DefaultPredictor

class DocumentClassifier:
    def __init__(self, model_path, device=None):
        """Initialize the document classifier using Detectron2.
        
        Args:
            model_path (str): Path to the Detectron2 model weights
            device (str, optional): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Configure Detectron2 model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15  # Number of classes from local_inference.py
        self.cfg.MODEL.DEVICE = self.device
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Same as local_inference.py
        
        # Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)
        print(f"Loaded Detectron2 model from {model_path}")
        
        # Class labels matching local_inference.py
        self.classes = [
            'bottom_left_corner', 'bottom_right_corner', 'citizenship_status',
            'country_of_birth', 'date_of_birth', 'face', 'id_document', 'id_number',
            'names', 'nationality', 'sex', 'signature', 'surname', 'top_left_corner',
            'top_right_corner'
        ]
    
    def classify(self, image_path):
        """Classify an ID document using Detectron2 detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            # Run inference
            outputs = self.predictor(image)
            
            # Get detected instances
            instances = outputs["instances"].to("cpu")
            
            # If no detections, return unknown
            if len(instances) == 0:
                return "unknown", 0.0
            
            # Get the most confident prediction
            scores = instances.scores
            classes = instances.pred_classes
            
            # Get index of highest confidence detection
            max_conf_idx = torch.argmax(scores)
            predicted_class = self.classes[classes[max_conf_idx]]
            confidence = scores[max_conf_idx].item()
            
            return predicted_class, confidence
            
        except Exception as e:
            raise RuntimeError(f"Classification failed for {image_path}: {str(e)}")
    
    def batch_classify(self, image_paths):
        """Classify multiple images.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of (predicted_class, confidence_score) tuples
        """
        results = []
        for path in image_paths:
            try:
                doc_type, confidence = self.classify(path)
                results.append((doc_type, confidence))
            except Exception as e:
                print(f"Warning: Failed to process {path}: {str(e)}")
                results.append(("unknown", 0.0))
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize classifier with model path from local_inference.py
    model_path = "models/model_final.pth"
    classifier = DocumentClassifier(model_path)
    
    # Single image classification
    image_path = "test_images/sample_id.jpg"
    if cv2.imread(image_path) is not None:
        predicted_class, confidence = classifier.classify(image_path)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
