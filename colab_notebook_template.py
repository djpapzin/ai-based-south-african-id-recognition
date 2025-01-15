# ID Card Detection - Training Notebook
# Copy each cell into a new Google Colab notebook

# Cell 1: Install dependencies
%%capture
!pip install torch torchvision
!pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Cell 2: Import required libraries and check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Cell 3: Mount Google Drive (if using)
from google.colab import drive
drive.mount('/content/drive')

# Cell 4: Download and prepare training script
%%writefile train_colab.py
import os
import json
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

def setup_cfg(train_json, val_json, num_classes, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("id_card_train",)
    cfg.DATASETS.TEST = ("id_card_val",)
    
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = output_dir
    
    return cfg

def train_model(train_json, val_json, train_images_dir, val_images_dir, output_dir, num_classes):
    print("Setting up datasets...")
    register_coco_instances(
        "id_card_train", 
        {}, 
        train_json, 
        train_images_dir
    )
    register_coco_instances(
        "id_card_val", 
        {}, 
        val_json, 
        val_images_dir
    )
    
    print("Configuring model...")
    cfg = setup_cfg(train_json, val_json, num_classes, output_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("Starting training...")
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# Cell 5: Upload dataset files
# Option 1: Upload through Colab interface
from google.colab import files
print("Please upload your dataset files (train.json, val.json, and images folder)")
uploaded = files.upload()

# Option 2: If files are in Google Drive, copy them to Colab workspace
# !cp /content/drive/MyDrive/path/to/dataset/* /content/dataset/

# Cell 6: Verify dataset structure
!ls -R /content/

# Cell 7: Run training
!python train_colab.py \
    --train-json "/content/train.json" \
    --val-json "/content/val.json" \
    --train-images-dir "/content/images" \
    --val-images-dir "/content/images" \
    --output-dir "/content/training_output" \
    --num-classes 7

# Cell 8: Monitor training progress
# This cell can be run periodically to check training metrics
!cat /content/training_output/metrics.json

# Cell 9: Save model to Drive (if using)
!cp -r /content/training_output /content/drive/MyDrive/id_card_model/ 