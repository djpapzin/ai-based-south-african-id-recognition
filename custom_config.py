from detectron2.config import get_cfg
from detectron2 import model_zoo

def get_custom_config():
    """
    Create a custom configuration for a model with two classes
    Dataset size: 458 total images (New: 301, Old: 157)
    """
    cfg = get_cfg()
    
    # Load the base configuration from model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Set the number of classes (2 classes: "New" and "Old")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    # Set training parameters
    cfg.DATASETS.TRAIN = ("custom_dataset",)
    cfg.DATASETS.TEST = ("custom_dataset",)
    
    # Adjusted training hyperparameters based on dataset size
    cfg.SOLVER.IMS_PER_BATCH = 4  # Increased batch size
    cfg.SOLVER.BASE_LR = 0.0025  # Slightly increased learning rate
    cfg.SOLVER.MAX_ITER = 2000  # Increased iterations for larger dataset
    cfg.SOLVER.STEPS = (1000, 1500)  # Learning rate decay steps
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Increased for better sampling
    
    # Set the score threshold for testing
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    # Add class weights to handle class imbalance (301 vs 157)
    cfg.MODEL.ROI_HEADS.LOSS_WEIGHT = 1.0
    
    return cfg

if __name__ == "__main__":
    cfg = get_custom_config()
    print("Configuration created with the following settings:")
    print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"Training dataset: {cfg.DATASETS.TRAIN}")
    print(f"Testing dataset: {cfg.DATASETS.TEST}")
    print(f"Images per batch: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"ROI batch size per image: {cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE}") 