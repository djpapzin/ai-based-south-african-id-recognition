import os
import datetime
import subprocess
import signal
import time
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
import torch

def kill_tensorboard():
    # Windows-specific way to kill processes on port 6006
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'tensorboard.exe'], capture_output=True)
    except:
        pass  # Ignore if no process exists

def start_tensorboard(log_dir):
    # Kill any existing TensorBoard
    kill_tensorboard()
    
    # Start TensorBoard in the background
    subprocess.Popen(['tensorboard', '--logdir', log_dir, '--port', '6006'])
    print("TensorBoard started. Visit http://localhost:6006 to view training progress")
    time.sleep(3)  # Give TensorBoard time to start

def main():
    # Setup logger
    setup_logger()

    # Register the datasets
    register_coco_instances(
        "id_train",
        {},
        "dj_object_detection_dataset/fixed_detectron2_coco.json",
        "dj_object_detection_dataset/images"
    )
    register_coco_instances(
        "id_train_abenathi",
        {},
        "abenathi_object_detection_dataset/fixed_detectron2_coco.json",
        "abenathi_object_detection_dataset/images"
    )

    # Create config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("id_train", "id_train_abenathi")
    cfg.DATASETS.TEST = ()
    
    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.MODEL.DEVICE}")

    # Model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    
    # Solver parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []
    
    # Set up output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.OUTPUT_DIR = f"output_{timestamp}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Start TensorBoard
    start_tensorboard(cfg.OUTPUT_DIR)
    
    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
