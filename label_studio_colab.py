# Title: Label Studio with ML Backend on Google Colab
# Save this as a .py file and upload to Colab, or copy contents to a Colab notebook

# Install required packages
!pip install label-studio label-studio-ml==1.0.9 torch torchvision detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

import os
import json
import torch
import numpy as np
from pathlib import Path
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

class IDCardDetector(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(IDCardDetector, self).__init__(**kwargs)
        
        # Load model
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # Adjust based on your number of classes
        self.cfg.MODEL.WEIGHTS = "path_to_your_model.pth"  # You'll need to update this
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.predictor = DefaultPredictor(self.cfg)
        
        # Define your label map
        self.label_map = {
            0: "New ID Card",
            1: "Old ID Book",
            2: "Names",
            3: "Surname",
            4: "ID Number",
            5: "Date of Birth",
            6: "Sex",
            7: "Nationality",
            8: "Country of Birth",
            9: "Face Photo",
            10: "Signature",
            11: "Citizenship Status"
        }
        
    def predict(self, tasks, **kwargs):
        predictions = []
        
        for task in tasks:
            image_path = task['data']['image']
            image = cv2.imread(image_path)
            
            # Get predictions
            outputs = self.predictor(image)
            
            # Convert predictions to Label Studio format
            pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
            pred_classes = outputs["instances"].pred_classes.cpu().numpy()
            scores = outputs["instances"].scores.cpu().numpy()
            
            results = []
            for box, class_id, score in zip(pred_boxes, pred_classes, scores):
                x1, y1, x2, y2 = box
                results.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [self.label_map[class_id]],
                        "x": float(x1 / image.shape[1] * 100),
                        "y": float(y1 / image.shape[0] * 100),
                        "width": float((x2 - x1) / image.shape[1] * 100),
                        "height": float((y2 - y1) / image.shape[0] * 100),
                        "score": float(score)
                    }
                })
            
            predictions.append({"result": results})
        
        return predictions

# Set up Label Studio
!label-studio start --init --force --no-browser --ml-backends http://localhost:9090

# In a separate cell, run the ML backend
!label-studio-ml init my_ml_backend
!label-studio-ml start my_ml_backend/

# Instructions for setting up project:
"""
1. Access Label Studio at the provided URL
2. Create a new project
3. Import your images
4. Set up labeling interface with your existing configuration
5. Connect ML backend in project settings
6. Start labeling with ML assistance
"""

# Export annotations (when needed)
!label-studio export {project_id} --format JSON --path ./annotations.json 