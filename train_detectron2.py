import os
import torch
import logging
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.events import TensorboardXWriter
from detectron2.utils.logger import setup_logger

class Trainer(DefaultTrainer):
    """Custom trainer class with evaluation hooks."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, TensorboardXWriter(self.cfg.OUTPUT_DIR))
        return hooks

def setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir, iterations=5000):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Dataset config
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    
    # Determine device
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        cfg.SOLVER.IMS_PER_BATCH = 2
    else:
        cfg.MODEL.DEVICE = "cpu"
        cfg.SOLVER.IMS_PER_BATCH = 1
    
    # Model config
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
    # Solver config
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = iterations
    cfg.SOLVER.STEPS = (iterations // 2, iterations * 3 // 4)
    cfg.SOLVER.GAMMA = 0.1
    
    # Test config
    cfg.TEST.EVAL_PERIOD = iterations // 10
    
    # Output config
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg

def register_datasets(train_path, val_path):
    """
    Register the datasets in COCO format.
    """
    register_coco_instances(
        "id_card_train",
        {},
        os.path.join(train_path, "annotations.json"),
        os.path.join(train_path, "images")
    )
    register_coco_instances(
        "id_card_val",
        {},
        os.path.join(val_path, "annotations.json"),
        os.path.join(val_path, "images")
    )
    
    return "id_card_train", "id_card_val"

def train(train_path, val_path, output_dir, num_classes, iterations=5000):
    """
    Main training function.
    """
    # Setup logger
    setup_logger(output_dir)
    logger = logging.getLogger("detectron2")
    
    # Register datasets
    train_dataset_name, val_dataset_name = register_datasets(train_path, val_path)
    
    # Setup config
    cfg = setup_cfg(train_dataset_name, val_dataset_name, num_classes, output_dir, iterations)
    
    # Print training info
    logger.info("Starting training with config:")
    logger.info(f"Device: {cfg.MODEL.DEVICE}")
    logger.info(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    logger.info(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    
    # Create trainer
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    logger.info("Training started...")
    trainer.train()
    
    return cfg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Detectron2 model on ID card dataset')
    parser.add_argument('--train-path', required=True, help='Path to training dataset directory')
    parser.add_argument('--val-path', required=True, help='Path to validation dataset directory')
    parser.add_argument('--output-dir', required=True, help='Path to output directory')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of training iterations')
    
    args = parser.parse_args()
    
    train(args.train_path, args.val_path, args.output_dir, args.num_classes, args.iterations) 