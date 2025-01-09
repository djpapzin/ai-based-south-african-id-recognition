import os
import torch
import cv2
import numpy as np
from typing import List, Dict
from label_studio_ml.model import LabelStudioMLBase
from segment_anything import SamPredictor, sam_model_registry

MODEL_TYPE = os.environ.get("MODEL_TYPE", "vit_h")
CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    "sam_vit_h_4b8939.pth",
)

DEVICE = os.environ.get("DEVICE", "cuda")

class SAMBackend(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # load model
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).to(DEVICE)
        self.predictor = SamPredictor(sam)
        
        print("SAM model initialized successfully...")

    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        predictions = []
        for task in tasks:
           # Fetch and process the input image from task data.
           image_path = self.get_first_tag_value(task, 'image')
           if not image_path:
               print("No image path found")
               continue
           
           image_path = self.resolve_uri(image_path)

           image = cv2.imread(image_path)
           if image is None:
             print(f"Unable to load image {image_path}")
             continue
           image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           self.predictor.set_image(image)
          
           original_h, original_w = image.shape[:2]
          
           input_points = np.array(
              [[original_w / 2, original_h / 2]]
             )
           input_labels = np.array([1])
           masks, _, _ = self.predictor.predict(
               point_coords=input_points,
               point_labels=input_labels,
               multimask_output=False,
           )
           # Process predictions to match Label Studio format
           
           mask = masks[0]
           mask = mask.astype(np.uint8) * 255

           mask_encoded = self._encode_mask(mask)
           x,y,w,h = self._get_bounding_box_from_mask(mask)
           
           
           predictions.append({
               "result": [{
                   "from_name": "label",
                   "to_name": "image",
                   "type": "masks",
                    "value":{
                       "format": "rle",
                       "rle": mask_encoded,
                       "box": [x,y,w,h]
                       }
                       
                   }],
               "score": 1.0,
           })
        return predictions

    def _encode_mask(self, mask: np.ndarray) -> str:
        """ Encodes a mask to RLE format to reduce size"""
        mask_fortran = np.asfortranarray(mask)
        return self.encode(mask_fortran)
    
    def _get_bounding_box_from_mask(self, mask: np.ndarray) -> str:
        """ Creates a bounding box from a mask
        """
        indices = np.argwhere(mask > 0)
        if len(indices) == 0:
          return 0, 0, 0, 0
        x_min = np.min(indices[:,1])
        x_max = np.max(indices[:,1])
        y_min = np.min(indices[:,0])
        y_max = np.max(indices[:,0])
        width = x_max - x_min
        height = y_max - y_min
        return x_min, y_min, width, height 