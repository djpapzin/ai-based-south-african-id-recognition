import os
import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import json
from pathlib import Path

class IDAnnotator:
    def __init__(self, model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth", device="cpu"):
        """Initialize the ID Annotator with SAM model"""
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        
    def process_image(self, image_path):
        """Process a single image and return annotations"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
            
        # Convert to RGB for SAM
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(image)
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Define points of interest (you can modify these)
        points = [
            (w/2, h/2),  # Center point
            (w/4, h/4),  # Top-left quadrant
            (3*w/4, h/4),  # Top-right quadrant
            (w/4, 3*h/4),  # Bottom-left quadrant
            (3*w/4, 3*h/4)  # Bottom-right quadrant
        ]
        
        annotations = []
        
        # Process each point
        for point in points:
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])
            
            masks, scores, _ = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Get the best mask
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # Get bounding box
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                
                annotations.append({
                    "bbox": [int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min)],
                    "score": float(scores[best_mask_idx]),
                    "point": [int(point[0]), int(point[1])]
                })
        
        return annotations

def main():
    # Initialize annotator
    annotator = IDAnnotator(device="cpu")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Process all images in IDcopies directory
    id_copies_dir = Path("IDcopies")
    results = {}
    
    for id_dir in id_copies_dir.iterdir():
        if id_dir.is_dir():
            id_number = id_dir.name
            results[id_number] = {}
            
            for image_file in id_dir.glob("*.pdf"):
                # Skip PDF files for now as they need conversion
                continue
                
            for image_file in id_dir.glob("*.[jJ][pP][eE][gG]"):
                image_name = image_file.name
                print(f"Processing {image_file}")
                
                annotations = annotator.process_image(str(image_file))
                if annotations:
                    results[id_number][image_name] = annotations
    
    # Save results
    with open(results_dir / "annotations.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 