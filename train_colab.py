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
from detectron2.engine import DefaultPredictor

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
    
    # Force CPU
    cfg.MODEL.DEVICE = "cpu"

    return cfg

def train_model(train_json, val_json, train_images_dir, val_images_dir, output_dir, num_classes):
    """
    Train Detectron2 model on Google Colab
    
    Args:
        train_json: Path to training annotations JSON
        val_json: Path to validation annotations JSON
        train_images_dir: Directory containing training images
        val_images_dir: Directory containing validation images
        output_dir: Directory to save model outputs
        num_classes: Number of object classes
    """
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
    
    print(f"""
Training completed! Output files are saved in: {output_dir}
- Model weights: model_final.pth
- Metrics: metrics.json
- Other logs and evaluation results are in the same directory
""")

def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Only 2 classes: New and Old

    model_path = f"{GDRIVE_PATH}/trained_model/training_output/model_final.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    # Force CPU
    cfg.MODEL.DEVICE = "cpu"

    return DefaultPredictor(cfg)

def process_ocr_on_detected_regions(base_dir="detected_regions", output_file="ocr_results.json"):
    """Process OCR on all detected regions and save results.
    
    Args:
        base_dir: Directory containing detected regions
        output_file: JSON file to save OCR results
    """
    results = {}
    
    # Walk through all subdirectories in the detected regions folder
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                
                # Get parent folder name (original image name)
                parent_folder = os.path.basename(os.path.dirname(image_path))
                
                # Perform OCR with preprocessing
                text = perform_ocr(image_path)
                
                # Store results
                if parent_folder not in results:
                    results[parent_folder] = []
                    
                results[parent_folder].append({
                    'region_file': file,
                    'text': text,
                    'full_path': image_path
                })
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"OCR results saved to {output_file}")
    return results

# Function to display OCR results
def display_ocr_results(results):
    """Display OCR results in a readable format."""
    for image_name, regions in results.items():
        print(f"\nResults for image: {image_name}")
        print("-" * 50)
        for region in regions:
            print(f"\nRegion: {region['region_file']}")
            print("Text:")
            print(region['text'] if region['text'] else "[No text detected]")
            print("-" * 30)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Detectron2 model on Google Colab')
    parser.add_argument('--train-json', required=True,
                      help='Path to training annotations JSON')
    parser.add_argument('--val-json', required=True,
                      help='Path to validation annotations JSON')
    parser.add_argument('--train-images-dir', required=True,
                      help='Directory containing training images')
    parser.add_argument('--val-images-dir', required=True,
                      help='Directory containing validation images')
    parser.add_argument('--output-dir', default='training_output',
                      help='Directory to save model outputs')
    parser.add_argument('--num-classes', type=int, required=True,
                      help='Number of object classes')
    
    args = parser.parse_args()
    
    train_model(
        train_json=args.train_json,
        val_json=args.val_json,
        train_images_dir=args.train_images_dir,
        val_images_dir=args.val_images_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes
    )

    # Add this after test_all_images() call
    print("\nProcessing OCR on detected regions...")
    ocr_results = process_ocr_on_detected_regions()
    display_ocr_results(ocr_results)

if __name__ == "__main__":
    main() 