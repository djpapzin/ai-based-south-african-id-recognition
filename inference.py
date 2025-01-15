from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
import torch
import cv2
import json
import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from detectron2_config import setup_cfg
from data_loader import register_datasets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class Predictor:
    """
    Class to handle model inference, visualization, and result export.
    """
    def __init__(
        self,
        model_weights: str,
        config_path: str,
        num_classes: int,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize predictor with model and configuration.
        
        Args:
            model_weights: Path to model weights
            config_path: Path to config file
            num_classes: Number of classes
            confidence_threshold: Confidence threshold for predictions
        """
        self.logger = logging.getLogger(__name__)
        
        # Create config
        self.cfg = setup_cfg(
            train_dataset_name="test_dataset",
            num_classes=num_classes,
            output_dir="./inference_output"
        )
        
        # Override config with inference settings
        self.cfg.MODEL.WEIGHTS = model_weights
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        
        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        # Get metadata
        self.metadata = MetadataCatalog.get("test_dataset")
        
        self.logger.info(f"Initialized predictor with model: {model_weights}")
        
    def predict_image(
        self,
        image_path: str,
        output_dir: str,
        save_visualization: bool = True
    ) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            save_visualization: Whether to save visualization
            
        Returns:
            Dict containing prediction results
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not read image: {image_path}")
            return None
            
        # Run inference
        predictions = self.predictor(image)
        
        # Get image name for output
        image_name = Path(image_path).stem
        
        # Process predictions
        instances = predictions["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Create results dictionary
        results = {
            "image_name": image_name,
            "image_path": image_path,
            "predictions": []
        }
        
        # Process each prediction
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.tolist()
            results["predictions"].append({
                "category_id": int(class_id),
                "category_name": self.metadata.thing_classes[class_id],
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # Convert to [x, y, width, height]
                "score": float(score),
                "bbox_mode": BoxMode.XYXY_ABS
            })
        
        # Save visualization if requested
        if save_visualization:
            vis_output = self._visualize_predictions(
                image,
                predictions,
                image_name,
                output_dir
            )
            results["visualization_path"] = vis_output
            
        return results
    
    def _visualize_predictions(
        self,
        image: np.ndarray,
        predictions: Dict,
        image_name: str,
        output_dir: str
    ) -> str:
        """
        Visualize predictions on image.
        
        Args:
            image: Input image
            predictions: Model predictions
            image_name: Name of the image
            output_dir: Output directory
            
        Returns:
            Path to saved visualization
        """
        # Create visualizer
        v = Visualizer(
            image[:, :, ::-1],
            metadata=self.metadata,
            scale=1.0
        )
        
        # Draw predictions
        vis_output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{image_name}_predictions.png")
        cv2.imwrite(output_path, vis_output.get_image()[:, :, ::-1])
        
        return output_path

def run_inference(
    model_weights: str,
    test_json: str,
    test_image_dir: str,
    output_dir: str,
    num_classes: int,
    confidence_threshold: float = 0.5
) -> None:
    """
    Run inference on test dataset and save results.
    
    Args:
        model_weights: Path to model weights
        test_json: Path to test set COCO JSON
        test_image_dir: Path to test images directory
        output_dir: Directory to save outputs
        num_classes: Number of classes
        confidence_threshold: Confidence threshold for predictions
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting inference...")
    
    # Register test dataset
    success = register_datasets(
        train_json=test_json,  # Use test set as train for registration
        train_image_dir=test_image_dir
    )
    
    if not success:
        logger.error("Failed to register test dataset")
        return
    
    # Create output directories
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize predictor
    predictor = Predictor(
        model_weights=model_weights,
        config_path=None,  # Using default config
        num_classes=num_classes,
        confidence_threshold=confidence_threshold
    )
    
    # Load test dataset
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    
    # Run inference on all images
    all_results = []
    for img in test_data['images']:
        image_path = os.path.join(test_image_dir, img['file_name'])
        logger.info(f"Processing image: {img['file_name']}")
        
        # Run prediction
        results = predictor.predict_image(
            image_path=image_path,
            output_dir=vis_dir,
            save_visualization=True
        )
        
        if results is not None:
            all_results.append(results)
    
    # Save all results to JSON
    output_json = os.path.join(output_dir, "inference_results.json")
    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"Inference completed. Results saved to {output_json}")
    logger.info(f"Visualizations saved to {vis_dir}")

def main():
    """Example usage of inference script."""
    run_inference(
        model_weights="path/to/model_final.pth",
        test_json="path/to/test.json",
        test_image_dir="path/to/test/images",
        output_dir="./inference_output",
        num_classes=5,
        confidence_threshold=0.5
    )

if __name__ == "__main__":
    main() 