from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.config import CfgNode as CN
from detectron2 import model_zoo
from detectron2.data import transforms as T
import torch
import copy
import numpy as np
from typing import List, Dict, Optional
import albumentations as A

class CustomTrainer(DefaultTrainer):
    """Custom trainer to override the default data loader."""
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build a custom data loader that includes advanced augmentations.
        """
        mapper = CustomDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

class CustomDatasetMapper(DatasetMapper):
    """Custom dataset mapper to apply advanced augmentations."""
    
    def __init__(self, cfg, is_train: bool = True):
        super().__init__(cfg, is_train)
        
        # Define advanced augmentation pipeline using albumentations
        self.transform = A.Compose([
            A.RandomCrop(width=cfg.INPUT.CROP.SIZE[0], height=cfg.INPUT.CROP.SIZE[1], p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.CLAHE(p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))
    
    def __call__(self, dataset_dict):
        """
        Apply custom transformations to the input data.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        
        if self.is_train:
            # Apply custom augmentations
            boxes = [obj["bbox"] for obj in dataset_dict["annotations"]]
            category_ids = [obj["category_id"] for obj in dataset_dict["annotations"]]
            
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                category_ids=category_ids
            )
            
            image = transformed["image"]
            boxes = transformed["bboxes"]
            category_ids = transformed["category_ids"]
            
            # Update annotations with transformed boxes
            for i, obj in enumerate(dataset_dict["annotations"]):
                obj["bbox"] = boxes[i]
                obj["category_id"] = category_ids[i]
        
        # Apply standard Detectron2 transformations
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        
        # Apply standard annotation transformations
        utils.transform_instance_annotations(
            dataset_dict["annotations"], transforms, image.shape[:2]
        )
        
        # Convert to Detectron2 format
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(
            dataset_dict["annotations"], image.shape[:2]
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        
        return dataset_dict

def setup_cfg(train_dataset_name: str,
              num_classes: int,
              output_dir: str) -> CN:
    """
    Create a Detectron2 configuration with custom settings.
    
    Args:
        train_dataset_name: Name of the training dataset
        num_classes: Number of object classes (including background)
        output_dir: Directory to save outputs
        
    Returns:
        cfg: Detectron2 configuration object
    """
    cfg = get_cfg()
    
    # Load Faster R-CNN base configuration
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Load pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    
    # Dataset settings
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = ()
    
    # Solver settings
    cfg.SOLVER.IMS_PER_BATCH = 4  # Batch size
    cfg.SOLVER.BASE_LR = 0.001  # Initial learning rate
    cfg.SOLVER.MAX_ITER = 10000  # Total iterations
    cfg.SOLVER.STEPS = (7000, 9000)  # Steps for LR decay
    cfg.SOLVER.GAMMA = 0.1  # LR decay factor
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every N iterations
    
    # Use AdamW optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.WEIGHT_DECAY = 0.01
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    
    # Gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    
    # Input settings
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)  # Random resize
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Crop settings
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    
    # Data augmentation settings
    cfg.INPUT.RANDOM_FLIP = "horizontal"
    
    # Model settings
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    # FPN settings
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    
    # RPN settings
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
    
    # Output directory
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

def main():
    """Example usage of the configuration."""
    # Set up configuration
    cfg = setup_cfg(
        train_dataset_name="my_dataset_train",
        num_classes=5,  # Adjust based on your number of classes
        output_dir="./output"
    )
    
    # Create trainer
    trainer = CustomTrainer(cfg)
    
    # Train model
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main() 